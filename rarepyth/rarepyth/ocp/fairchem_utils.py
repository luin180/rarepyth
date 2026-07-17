
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:37:14 2025

@author: Wang Junhao
"""

import os
import re
import copy
import logging
import datetime
import warnings
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm
import ase
from ase import Atom, Atoms
from ase.constraints import FixAtoms, Hookean
from ase.phonons import Phonons
from ase.io.vasp import write_vasp
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE2
from ase.optimize.minimahopping import MinimaHopping
from pymatgen.core import Structure

from fairchem.core.common.registry import registry
from fairchem.core.common import distutils
from fairchem.core.common.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.common.relaxation.ase_utils import OCPCalculator

from rarepyth.base import _mswindows
if not _mswindows:
    from rarepyth.base import runcmd

# For fairchem-core 1.10.0
# Lack support for flag args

setup_imports()
setup_logging()
warnings.simplefilter(action='ignore', category=FutureWarning)


class FairchemJob():
    def __init__(
        self,
        config,
    ):
        self.config = copy.deepcopy(config)
        self.config["trainer"] = config.get("trainer", "ocp")
        if "model_attributes" in config:
            self.config["model_attributes"]["name"] = self.config.pop("model")
            self.config["model"] = self.config["model_attributes"]
        if not self.config.get("loss_functions"):
            self.config = update_config(self.config)
        self.config["trainer"] = self.config.get("trainer", "ocp")
        if self.config["trainer"] in ["forces", "equiformerv2_forces"]:
            self.task_name = "s2ef"
        elif self.config["trainer"] in ["energy", "equiformerv2_energy"]:
            self.task_name = "is2re"
        else:
            raise Exception(
                f'Trainer "{self.config["trainer"]}" not supported yet.')
        if _mswindows:
            self.config["optim"]["num_workers"] = 0

        if torch.cuda.is_available() and not self.config.get("cpu", False):
            local_rank = self.config.get("local_rank", 0)
            logging.info(f"local rank base: {local_rank}")
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
            self.config['cpu'] = True
        now = datetime.datetime.now().timestamp()
        timestamp_tensor = torch.tensor(now, dtype=torch.float64).to(device)
        # Create directories from master rank only
        distutils.broadcast(timestamp_tensor, 0)
        timestamp_str = datetime.datetime.fromtimestamp(
            timestamp_tensor.item()
        ).strftime("%Y-%m-%d-%H-%M-%S")
        if self.config.get("identifier", ""):
            timestamp_str += "-" + self.config.get("identifier", "")

        self.trainer = registry.get_trainer_class(self.config["trainer"])(
            task=self.config["task"],
            model=self.config["model"],
            outputs=self.config["outputs"],
            dataset=[{}, {}, {}],
            optimizer=self.config["optim"],
            loss_functions=self.config["loss_functions"],
            evaluation_metrics=self.config["evaluation_metrics"],
            identifier=self.config.get("identifier", ""),
            timestamp_id=timestamp_str,
            run_dir=self.config.get("run_dir", './'),
            is_debug=self.config.get("is_debug", False),
            print_every=self.config.get("print_every", 10),
            seed=self.config.get("seed", 0),
            logger="tensorboard",
            local_rank=self.config.get("local_rank", 0),
            amp=self.config.get("amp", False),
            cpu=self.config.get("cpu", False),
            name=self.task_name,
            gp_gpus=self.config.get("gp_gpus", None),
        )
        self.checkpoint_path = self.config.get("checkpoint", None)
        if self.checkpoint_path:
            if self.config['trainer'] == 'equiformerv2_forces':
                self.load_checkpoint_delay = True
            else:
                self.load_checkpoint(self.checkpoint_path)

    @classmethod
    def from_yml(cls, yml_path):
        config, duplicates_warning, duplicates_error = load_config(yml_path)
        if len(duplicates_warning) > 0:
            logging.warning(
                f"Overwritten config parameters from included configs "
                f"(non-included parameters take precedence): {duplicates_warning}"
            )
        if len(duplicates_error) > 0:
            raise ValueError(
                f"Conflicting (duplicate) parameters in simultaneously "
                f"included configs: {duplicates_error}"
            )
        return cls(config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, overwrite={}):
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"))
        config = update_config(checkpoint["config"])
        config['checkpoint'] = checkpoint_path
        cls.overwrite_config(config, overwrite)
        if config.get('trainer', None) == 'is2re':
            if ((isinstance(config['model'], str)
                 and 'equiformer' in config['model'])
                or (isinstance(config['model'], dict)
                    and 'equiformer' in config['model']['name'])):
                config['trainer'] = 'equiformerv2_energy'
            else:
                config['trainer'] = 'energy'
        elif config.get('trainer', None) == 's2ef':
            if ((isinstance(config['model'], str)
                 and 'equiformer' in config['model'])
                or (isinstance(config['model'], dict)
                    and 'equiformer' in config['model']['name'])):
                config['trainer'] = 'equiformerv2_forces'
            else:
                config['trainer'] = 'forces'
        return cls(config)

    @staticmethod
    def overwrite_config(config, overwrite):
        for key, value in overwrite.items():
            if key in config.keys():
                if value is None:
                    del config[key]
                elif type(config[key]) is not type(value):
                    raise TypeError(
                        f"Type mismatch for '{key}'. Original is {type(config[key]).__name__}, overwrite is {type(value).__name__}."
                    )
                elif isinstance(config[key], dict):
                    FairchemJob.overwrite_config(config[key], value)
                else:
                    config[key] = value
            else:
                if isinstance(value, dict):
                    config[key] = {}
                    FairchemJob.overwrite_config(config[key], value)
                elif value is not None:
                    config[key] = value

    def load_checkpoint(self, checkpoint_path):
        self.config["checkpoint"] = checkpoint_path
        self.checkpoint_path = checkpoint_path
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def load_datasets(
        self,
        train_set_path=None,
        valid_set_path=None,
        test_set_path=None,
        keep_normalizer=True
    ):
        # Read datasets in config
        if 'src' in self.config['dataset'].keys():
            self.trainer.config['dataset'] = self.config.get('dataset')
            self.trainer.config['val_dataset'] = self.config.get(
                'val_dataset', {})
            self.trainer.config['test_dataset'] = self.config.get(
                'test_dataset', {})
        else:
            self.trainer.config['dataset'] = self.config['dataset'].get(
                'train', {})
            self.trainer.config['val_dataset'] = self.config['dataset'].get(
                'val', {})
            self.trainer.config['test_dataset'] = self.config['dataset'].get(
                'test', {})
        for key in {'dataset', 'val_dataset', 'test_dataset'}:
            if not self.trainer.config[key]:
                self.trainer.config[key] = {}

        # Calculate mean and std for new datasets and attach them
        if train_set_path:
            self.trainer.config['dataset']['src'] = train_set_path
            if not keep_normalizer:
                self.trainer.config['dataset']['transforms'] = {
                    'normalizer': {}}
                lmdb_set = LmdbDataset({'src': train_set_path})
                energy_attr = next(attr for attr in [
                                   'y', 'y_relaxed', 'energy'] if hasattr(lmdb_set[0], attr))
                force_attr = next(attr for attr in [
                                  'force', 'forces'] if hasattr(lmdb_set[0], attr))
                energies = np.zeros([len(lmdb_set)], dtype='float64')
                if self.task_name == 'is2re':
                    for idx, data in tqdm(enumerate(lmdb_set)):
                        energies[idx] = getattr(data, energy_attr)

                elif self.task_name == 's2ef':
                    forces = []
                    for idx, data in tqdm(enumerate(lmdb_set)):
                        energies[idx] = getattr(data, energy_attr)
                        forces.append(getattr(data, force_attr))
                    target_vector = torch.cat(forces, dim=0)
                    self.trainer.config['dataset']['grad_target_mean'] = target_vector.mean(
                    ).item()
                    self.trainer.config['dataset']['grad_target_std'] = (
                        ((target_vector - target_vector.mean()) ** 2).sum()
                        / (len(target_vector) - 1).sqrt()).item()
                    self.trainer.config['dataset']['transforms']['normalizer']['forces'] = {
                        'mean': self.trainer.config['dataset']['grad_target_mean'],
                        'stdev': self.trainer.config['dataset']['grad_target_std']}

                self.trainer.config['dataset']['target_mean'] = np.mean(
                    energies)
                self.trainer.config['dataset']['target_std'] = np.std(energies)
                self.trainer.config['dataset']['transforms']['normalizer']['energy'] = {
                    'mean': self.trainer.config['dataset']['target_mean'],
                    'stdev': self.trainer.config['dataset']['target_std']}

        if valid_set_path:
            self.trainer.config['val_dataset']['src'] = valid_set_path
        if test_set_path:
            self.trainer.config['test_dataset']['src'] = test_set_path

        # Check whether the given datasets are available
        for key in {'dataset', 'val_dataset', 'test_dataset'}:
            if self.trainer.config[key] and not os.path.exists(self.trainer.config[key]['src']):
                logging.warning(f"Given {key} path '{self.trainer.config[key]['src']}' is not available, skip loading.")
                del self.trainer.config[key]['src']

        self.trainer.load_datasets()
        self.trainer.load_references_and_normalizers()
        self.trainer.load_extras()

        if self.checkpoint_path and self.load_checkpoint_delay:
            self.load_checkpoint(self.checkpoint_path)

    def train(
        self,
        disable_eval_tqdm=False,
        break_when_achieve_best=True,
        do_extra_epoches=True,
        run_test=True,
        best_epoch_starter=5,
        best_epoch_criteria=10
    ):
        """
        An enhanced version of fairchem OCPTrainer.train() method.
        """

        ensure_fitted(self.trainer._unwrapped_model, warn=True)

        eval_every = self.trainer.config["optim"].get(
            "eval_every", len(self.trainer.train_loader))
        if not eval_every == len(self.trainer.train_loader):
            logging.warning(
                'You have specified "eval_every" parameter manully. Auto breaking may cause unexpected results.')
        checkpoint_every = eval_every
        primary_metric = self.trainer.evaluation_metrics.get(
            "primary_metric", self.trainer.evaluator.task_primary_metric[self.trainer.name]
        )
        if not hasattr(self.trainer, "primary_metric") or self.trainer.primary_metric != primary_metric:
            self.trainer.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.trainer.primary_metric
        self.trainer.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.trainer.step // len(self.trainer.train_loader)

        self.best_val_metric_epoch = start_epoch

        if do_extra_epoches:
            max_epoches = self.trainer.config["optim"]["max_epochs"] + start_epoch

        for epoch_int in range(start_epoch, max_epoches):
            skip_steps = self.trainer.step % len(self.trainer.train_loader)
            self.trainer.train_sampler.set_epoch_and_start_iteration(
                epoch_int, skip_steps)
            train_loader_iter = iter(self.trainer.train_loader)

            for i in range(skip_steps, len(self.trainer.train_loader)):
                self.trainer.epoch = epoch_int + \
                    (i + 1) / len(self.trainer.train_loader)
                self.trainer.step = epoch_int * \
                    len(self.trainer.train_loader) + i + 1
                self.trainer.model.train()

                # Get a batch.
                batch = next(train_loader_iter)
                # Forward, loss, backward.
                with torch.autocast("cuda", enabled=self.trainer.scaler is not None):
                    out = self.trainer._forward(batch)
                    loss = self.trainer._compute_loss(out, batch)

                # Compute metrics.
                self.trainer.metrics = self.trainer._compute_metrics(
                    out,
                    batch,
                    self.trainer.evaluator,
                    self.trainer.metrics,
                )
                self.trainer.metrics = self.trainer.evaluator.update(
                    "loss", loss.item(), self.trainer.metrics)

                loss = self.trainer.scaler.scale(
                    loss) if self.trainer.scaler else loss
                self.trainer._backward(loss)

                # Log metrics.
                # log_dict = {k: self.trainer.metrics[k]["metric"] for k in self.trainer.metrics}
                # log_dict.update(
                #    {
                #        "lr": self.trainer.scheduler.get_lr(),
                #        "epoch": self.trainer.epoch,
                #        "step": self.trainer.step,
                #    }
                # )
                # if (
                #    self.trainer.step % self.trainer.config["cmd"]["print_every"] == 0
                #    and distutils.is_master()
                # ):
                #    log_str = [f"{k}: {v:.2e}" for k, v in log_dict.items()]
                #    logging.info(", ".join(log_str))
                #    self.trainer.metrics = {}

                # if self.trainer.logger is not None:
                #    self.trainer.logger.log(
                #        log_dict,
                #        step=self.trainer.step,
                #        split="train",
                #    )

                if checkpoint_every != -1 and self.trainer.step % checkpoint_every == 0:
                    self.trainer.save(
                        checkpoint_file="checkpoint.pt", training_state=True)

                # Evaluate on val set every `eval_every` iterations.
                if self.trainer.step % eval_every == 0:
                    if self.trainer.val_loader is not None:
                        val_metrics = self.trainer.validate(
                            split="val",
                            disable_tqdm=True,
                        )
                        if (
                            "mae" in primary_metric
                            and val_metrics[primary_metric]["metric"] < self.trainer.best_val_metric
                        ) or (
                            "mae" not in primary_metric
                            and val_metrics[primary_metric]["metric"] > self.trainer.best_val_metric
                        ):
                            self.best_val_metric_epoch = epoch_int
                            self.trainer.best_val_metric = val_metrics[primary_metric]["metric"]
                            self.trainer.save(
                                metrics=val_metrics,
                                checkpoint_file="best_checkpoint.pt",
                                training_state=False,
                            )
                            self.checkpoint_path = os.path.join(
                                self.trainer.config["cmd"]["checkpoint_dir"], "best_checkpoint.pt")
                            if run_test and self.trainer.test_loader is not None:
                                self.trainer.predict(
                                    self.trainer.test_loader,
                                    results_file="predictions",
                                    disable_tqdm=disable_eval_tqdm,
                                )

                    if self.trainer.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.trainer.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.trainer.run_relaxations()

                if self.trainer.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.trainer.step % eval_every == 0:
                        self.trainer.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.trainer.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.trainer.save(
                    checkpoint_file="checkpoint.pt", training_state=True)

            # Track best evaluation to get the best checkpoint
            if break_when_achieve_best and epoch_int > best_epoch_starter:
                if not self.best_val_metric_epoch:
                    logging.warning(
                        "The model didn't come better compared with checkpoint during the training.")
                    if not self.checkpoint_path:
                        self.checkpoint_path = os.path.join(
                            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt")
                    break
                if epoch_int - self.best_val_metric_epoch > best_epoch_criteria:
                    logging.info(
                        'Reached best_epoch_criteria, aborting the iteration.')
                    break
        else:
            if break_when_achieve_best:
                logging.warning(
                    "Best_epoch_criteria was never reached. This might because your epoches are too few."
                )

    def evaluate(self, disable_tqdm=False):
        """
        An combined version of fairchem OCPTrainer.predict() and BaseTrainer.validate() method, designed for active learning approach.
        """
        assert (self.checkpoint_path), 'Please train or load a model before evaluation.'

        data_sample = self.trainer.test_loader.dataset[0]
        for loss_fn in self.trainer.loss_functions:
            target_name, loss_info = loss_fn
            if not hasattr(data_sample, target_name):
                validate = False
                logging.warning(
                    "The given test dataset has no label. No validate metrics will be calculated.")
                break
        else:
            validate = True

        ensure_fitted(self.trainer._unwrapped_model, warn=True)
        if distutils.is_master() and not disable_tqdm:
            logging.info("Evaluating on test.")

        if validate:
            metrics = {}
            evaluator = Evaluator(
                task=self.trainer.name,
                eval_metrics=self.trainer.evaluation_metrics.get(
                    "metrics", Evaluator.task_metrics.get(
                        self.trainer.name, {})
                ),
            )

        rank = distutils.get_rank()

        self.trainer.model.eval()
        if self.trainer.ema is not None:
            self.trainer.ema.store()
            self.trainer.ema.copy_to()

        predictions = defaultdict(list)

        for _, batch in tqdm(
            enumerate(self.trainer.test_loader),
            total=len(self.trainer.test_loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.autocast("cuda", enabled=self.trainer.scaler is not None):
                batch.to(self.trainer.device)
                out = self.trainer._forward(batch)
            if validate:
                loss = self.trainer._compute_loss(out, batch)

            for target_key in self.trainer.config["outputs"]:
                pred = self.trainer._denorm_preds(
                    target_key, out[target_key], batch)
                if (
                    self.trainer.config["outputs"][target_key].get(
                        "prediction_dtype", "float16"
                    )
                    == "float32"
                    or self.trainer.config["task"].get("prediction_dtype", "float16")
                    == "float32"
                    or self.trainer.config["task"].get("dataset", "lmdb") == "oc22_lmdb"
                ):
                    dtype = torch.float32
                else:
                    dtype = torch.float16

                pred = pred.detach().cpu().to(dtype)

                if self.trainer.config["outputs"][target_key]["level"] == "atom":
                    batch_natoms = batch.natoms
                    batch_fixed = batch.fixed
                    per_image_pred = torch.split(pred, batch_natoms.tolist())

                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_preds = [
                        _pred[(fixed == 0).tolist()].numpy()
                        for _pred, fixed in zip(per_image_pred, _per_image_fixed)
                    ]
                    _chunk_idx = np.array(
                        [free_pred.shape[0]
                            for free_pred in _per_image_free_preds]
                    )
                    per_image_pred = _per_image_free_preds
                else:
                    per_image_pred = pred.numpy()
                    _chunk_idx = None

                predictions[f"{target_key}"].extend(per_image_pred)
                if _chunk_idx is not None:
                    if target_key == "forces":
                        predictions["chunk_idx"].extend(_chunk_idx)
                    else:
                        predictions[f"{target_key}_chunk_idx"].extend(
                            _chunk_idx)

            sids = (
                batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid
            )
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

            if validate:
                metrics = self.trainer._compute_metrics(
                    out, batch, evaluator, metrics)
                metrics = evaluator.update("loss", loss.item(), metrics)

            torch.cuda.empty_cache()

        if validate:
            metrics = self.trainer._aggregate_metrics(metrics)
            for k in metrics:
                predictions[k] = [metrics[k]["metric"]]

        keys = predictions.keys()
        results = distutils.gather_objects(predictions)
        distutils.synchronize()
        if distutils.is_master():
            gather_results = {
                key: list(chain(*(result[key] for result in results))) for key in keys
            }

            _, idx = np.unique(gather_results["ids"], return_index=True)
            for k in keys:
                if len(gather_results[k]) < len(idx):
                    gather_results[k] = np.array(gather_results[k])
                elif "chunk_idx" in k:
                    gather_results[k] = np.cumsum([gather_results[k][i] for i in idx])[
                        :-1
                    ]
                else:
                    if f"{k}_chunk_idx" in keys or k == "forces":
                        gather_results[k] = np.concatenate(
                            [gather_results[k][i] for i in idx]
                        )
                    else:
                        gather_results[k] = np.array(
                            [gather_results[k][i] for i in idx]
                        )

            self.result_path = os.path.join(
                self.trainer.config["cmd"]["results_dir"], f"{self.trainer.name}_evaluation.npz"
            )
            logging.info(f"Writing results to {self.result_path}")
            np.savez_compressed(self.result_path, **gather_results)

        if self.trainer.ema:
            self.trainer.ema.restore()

        return predictions

    def predict(self, disable_tqdm=False):
        assert (self.checkpoint_path), 'Please train or load a model before prediction.'
        predictions = self.trainer.predict(self.trainer.test_loader,
                                           results_file="predictions",
                                           disable_tqdm=disable_tqdm)
        return predictions


class FairchemThermo():
    def __init__(
        self,
        config_yml=None,
        checkpoint_path=None,
        model_name=None,
        local_cache=None,
        cpu=False,
        seed=0,
        disable_amp=True,
    ):
        trainer = 'forces'
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            config = update_config(checkpoint["config"])
            if ((isinstance(config['model'], str)
                 and 'equiformer' in config['model'])
                or (isinstance(config['model'], dict)
                    and 'equiformer' in config['model']['name'])):
                trainer = 'equiformerv2_forces'

        self.calc = OCPCalculator(config_yml=config_yml,
                                  checkpoint_path=checkpoint_path,
                                  model_name=model_name,
                                  local_cache=local_cache,
                                  trainer=trainer,
                                  cpu=cpu,
                                  seed=seed,
                                  disable_amp=disable_amp)
        self.structures = []
        self.dynamic_matrices = []
        self.sid_list = []
        self.fid_list = []

    def load_structure(self, filename, adsorbate_idx=None):
        self.structures.append(ase.io.read(filename))
        self.sid_list.append(None)
        self.fid_list.append(None)
        if adsorbate_idx:
            self.structures[-1].constraints = [FixAtoms(
                indices=list(set(range(len(self.structures[-1]))) -
                             set(adsorbate_idx)))]

    def load_datasets(self, test_set_path):
        for data in LmdbDataset({'src': test_set_path}):
            list_atoms = []
            for idx, atom in enumerate(data['atomic_numbers']):
                list_atoms.append(Atom(int(atom), data['pos'][idx]))
            self.structures.append(Atoms(symbols=list_atoms,
                                         cell=data['cell'].tolist()[0],
                                         tags=data['tags'].tolist(),
                                         pbc=True))
            self.sid_list.append(data['sid'])
            self.fid_list.append(data['fid'])
            constraint_list = []
            for idx, tag in enumerate(self.structures[-1].get_tags()):
                if tag == 0 or tag == 1:
                    constraint_list.append(idx)
            self.structures[-1].constraints = [FixAtoms(indices=constraint_list)]

    def export_displacement_structures(self, delta=0.015):
        for idx, struct in enumerate(self.structures):
            atoms_N = struct.copy()
            indices = list(set(range(len(struct))) -
                           set((struct.constraints[0].index)))
            pos = struct.positions.copy()
            for a in indices:
                for i in range(3):
                    for sign in [-1, 1]:
                        atoms_N.positions[a, i] = pos[a, i] + sign * delta
                        write_vasp(f'struct{idx}_{a}_{i}_{sign}.vasp', atoms_N)
                        atoms_N.positions[a, i] = pos[a, i]

    def calc_phonons(self, delta=0.015):
        self.dynamic_matrices = [None] * len(self.structures)
        for idx, struct in enumerate(self.structures):
            ph = Phonons(struct, calc=self.calc, delta=delta)
            ph.clean()
            free_atoms = list(set(range(len(struct))) -
                              set((struct.constraints[0].index)))
            if free_atoms:
                ph.set_atoms(free_atoms)
                ph.run()
                ph.read(acoustic=True)
                self.dynamic_matrices[idx] = ph.D_N[0].copy()
                ph.clean()
            else:
                self.dynamic_matrices[idx] = None

    def write_fake_vasp_output(self, idx, filename=None):
        try:
            poscar_filename = filename[0]
            outcar_filename = filename[1]
        except TypeError:
            poscar_filename = 'POSCAR'
            outcar_filename = 'OUTCAR'
        write_vasp(poscar_filename, self.structures[idx])
        eigenvalue, featurevector = np.linalg.eigh(
            0.5 * (self.dynamic_matrices[idx] + np.conj(self.dynamic_matrices[idx].T)))
        eigenvalue = -np.sort(-eigenvalue)
        eig_sign = np.sign(eigenvalue)
        eig_root = np.sqrt(eig_sign * eigenvalue)
        with open(outcar_filename, 'w') as file:
            for num, mode in enumerate(eig_root):
                if eig_sign[num] > 0:
                    file.write(f'{num + 1} f  =   {"%.6f" % (15.628 * mode)} THz   {"%.6f" % (31.256 * mode * np.pi)} 2PiTHz   {"%.6f" % (521.38 * mode)} cm-1   {"%.6f" % (64.64 * mode)} meV\n')
                else:
                    file.write(f'{num + 1} f/i=   {"%.6f" % (15.628 * mode)} THz   {"%.6f" % (31.256 * mode * np.pi)} 2PiTHz   {"%.6f" % (521.38 * mode)} cm-1   {"%.6f" % (64.64 * mode)} meV\n')

    if not _mswindows:
        def calc_corrections(self, temprature=298):
            if not len(self.structures) == len(self.dynamic_matrices):
                self.calc_phonons()
            self.thermal_corrections = [None] * len(self.structures)
            for idx, struct in enumerate(self.structures):
                if not self.dynamic_matrices[idx] is None:
                    self.write_fake_vasp_output(idx)
                    try:
                        self.thermal_corrections[idx] = float(re.findall(r'(-?\d+.\d*)\seV', runcmd(
                            f"(echo 501; echo {temprature}) | vaspkit | grep 'to G(T)'"))[0])
                    except Exception as e:
                        print(f'A error occoured while calculating corrections of {self.sid_list[idx]}:\n' + str(e))
                        self.thermal_corrections[idx] = 0.0
                else:
                    self.thermal_corrections[idx] = 0.0


class FairchemOptimizer():
    def __init__(
        self,
        config_yml=None,
        checkpoint_path=None,
        model_name=None,
        local_cache=None,
        cpu=False,
        seed=0,
        disable_amp=True,
    ):
        trainer = 'forces'
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            config = update_config(checkpoint["config"])
            if ((isinstance(config['model'], str)
                 and 'equiformer' in config['model'])
                or (isinstance(config['model'], dict)
                    and 'equiformer' in config['model']['name'])):
                trainer = 'equiformerv2_forces'

        self.calc = OCPCalculator(config_yml=config_yml,
                                  checkpoint_path=checkpoint_path,
                                  model_name=model_name,
                                  local_cache=local_cache,
                                  trainer=trainer,
                                  cpu=cpu,
                                  seed=seed,
                                  disable_amp=disable_amp)
        self.structures = []

    def load_structure(self, filename):
        self.structures.append(ase.io.read(filename))
        struct = Structure.from_file(filename)
        constraint = []
        for idx in range(struct.num_sites):
            if False in struct.sites[idx].properties['selective_dynamics']:
                constraint.append(idx)
        self.structures[-1].constraints = [FixAtoms(indices=constraint)]

    def calc_relaxation(self, forces_criteria=0.05, max_steps=300):
        energies = []
        for idx, structure in enumerate(self.structures):
            structure.calc = self.calc
            opt = BFGS(structure)
            opt.run(fmax=forces_criteria, steps=max_steps)
            energies.append(structure.get_potential_energy())
            write_vasp(f'POSCAR_RELAXED{idx}.vasp', structure)
        print(f'Energies:{energies}')


class AdjustedMinimaHopping(MinimaHopping):
    """Added a parameter to controll the maximum optimize steps during the
    optimization."""

    def __call__(self, totalsteps=None, maxtemp=None, max_optimize_steps=None):
        if max_optimize_steps is None:
            self.max_optimize_steps = 500
        else:
            self.max_optimize_steps = max_optimize_steps
        self._startup()
        while True:
            if (totalsteps and self._counter >= totalsteps):
                self._log('msg', 'Run terminated. Step #%i reached of '
                          '%i allowed. Increase totalsteps if resuming.'
                          % (self._counter, totalsteps))
                return
            if (maxtemp and self._temperature >= maxtemp):
                self._log('msg', 'Run terminated. Temperature is %.2f K;'
                          ' max temperature allowed %.2f K.'
                          % (self._temperature, maxtemp))
                return

            self._previous_optimum = self._atoms.copy()
            self._previous_energy = self._atoms.get_potential_energy()
            self._molecular_dynamics()
            self._optimize()
            self._counter += 1
            self._check_results()

    def _optimize(self):
        self._atoms.set_momenta(np.zeros(self._atoms.get_momenta().shape))
        with self._optimizer(self._atoms,
                             trajectory='qn%05i.traj' % self._counter,
                             logfile='qn%05i.log' % self._counter) as opt:
            self._log('msg', 'Optimization: qn%05i' % self._counter)
            opt.run(fmax=self._fmax, steps=self.max_optimize_steps)
            self._log('ene')


class FairchemGlobalOptimizer():
    def __init__(
        self,
        config_yml=None,
        checkpoint_path=None,
        model_name=None,
        local_cache=None,
        cpu=False,
        seed=0,
        disable_amp=True,
    ):
        trainer = 'forces'
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            config = update_config(checkpoint["config"])
            if ((isinstance(config['model'], str)
                 and 'equiformer' in config['model'])
                or (isinstance(config['model'], dict)
                    and 'equiformer' in config['model']['name'])):
                trainer = 'equiformerv2_forces'

        self.calc = OCPCalculator(config_yml=config_yml,
                                  checkpoint_path=checkpoint_path,
                                  model_name=model_name,
                                  local_cache=local_cache,
                                  trainer=trainer,
                                  cpu=cpu,
                                  seed=seed,
                                  disable_amp=disable_amp)

        self.constraints = []
        self.hookean_constraints = []

    def load_structure(self, filename):
        self.structure = ase.io.read(filename)
        struct = Structure.from_file(filename)
        constraint = []
        for idx in range(struct.num_sites):
            if 'selective_dynamics' in struct.sites[idx].properties.keys():
                if False in struct.sites[idx].properties['selective_dynamics']:
                    constraint.append(idx)
        self.constraints = [FixAtoms(indices=constraint)]
        self.structure.calc = self.calc

    def set_constraint_by_tagfile(self, tagfile='TAG', fixed_tags=[0]):
        tags = [0] * len(self.structure)
        constraint = []
        with open(tagfile, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                tags[int(line.split()[0]) - 1] = int(line.split()[1])
        for idx in range(len(self.structure)):
            if tags[idx] in fixed_tags:
                constraint.append(idx)
        self.constraints = [FixAtoms(indices=constraint)]

    def set_hookean_constraint_by_tagfile(self, tagfile='TAG', adsorbate_tag=2):
        """Current: C-H, C-C, C-O, O-H"""
        adsorbate_atom_idx = []
        with open(tagfile, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                if int(line.split()[1]) == adsorbate_tag:
                    adsorbate_atom_idx.append(int(line.split()[0]) - 1)
        for i, idx in enumerate(adsorbate_atom_idx[:-1]):
            i_atom = self.structure[idx]
            for jdx in adsorbate_atom_idx[i + 1:]:
                j_atom = self.structure[jdx]
                if (i_atom.number == 6 and j_atom.number == 1) or (i_atom.number == 1 and j_atom.number == 6):
                    if np.linalg.norm(i_atom.position - j_atom.position) <= 1.15:
                        self.hookean_constraints.append(Hookean(idx, jdx, 20.0, 1.15))
                if i_atom.number == 6 and j_atom.number == 6:
                    if np.linalg.norm(i_atom.position - j_atom.position) <= 1.80:
                        self.hookean_constraints.append(Hookean(idx, jdx, 20.0, 1.65))
                if (i_atom.number == 6 and j_atom.number == 8) or (i_atom.number == 8 and j_atom.number == 6):
                    if np.linalg.norm(i_atom.position - j_atom.position) <= 1.87:
                        self.hookean_constraints.append(Hookean(idx, jdx, 20.0, 1.72))
                if (i_atom.number == 8 and j_atom.number == 1) or (i_atom.number == 1 and j_atom.number == 8):
                    if np.linalg.norm(i_atom.position - j_atom.position) <= 1.10:
                        self.hookean_constraints.append(Hookean(idx, jdx, 20.0, 1.10))

    def load_hookean_constraints(self, filename='HOOKEAN'):
        """

        Hookean file format:
            8
            9
            15.0
            2.6

            8
            0.0 0.0 1.0 -15.0
            15.0
        This equals to [Hookean(7,8,15.0,rt=2.6), Hookean(7,(0.0,0.0,1.0,-15.0),15.0)]

        """
        with open(filename, 'r') as file:
            content = file.read()
        contents = content.strip().split('\n\n')
        for block in contents:
            blocks = block.split('\n')
            if len(blocks) == 3:
                a1 = int(blocks[0]) - 1
                k = float(blocks[2])
                a2_line = blocks[1].split()
                if len(a2_line) == 1:
                    a2 = int(a2_line[0]) - 1
                elif len(a2_line) == 3 or len(a2_line) == 4:
                    a2 = []
                    for value in a2_line:
                        a2.append(float(value))
                self.hookean_constraints.append(Hookean(a1, a2, k))
            elif len(blocks) == 4:
                a1 = int(blocks[0]) - 1
                k = float(blocks[2])
                rt = float(blocks[3])
                a2_line = blocks[1].split()
                if len(a2_line) == 1:
                    a2 = int(a2_line[0]) - 1
                elif len(a2_line) == 3 or len(a2_line) == 4:
                    a2 = []
                    for value in a2_line:
                        a2.append(float(value))
                self.hookean_constraints.append(Hookean(a1, a2, k, rt=rt))

    def run_minima_hopping(self, totalsteps=10, max_optimize_steps=500, optimizer='BFGS'):
        if os.path.exists('hop.log'):
            os.remove('hop.log')
        if os.path.exists('minima.traj'):
            os.remove('minima.traj')
        patterns = [re.compile(r'^md\d{5}\.log$'), re.compile(r'^md\d{5}\.traj$'),
                    re.compile(r'^qn\d{5}\.log$'), re.compile(r'^qn\d{5}\.traj$')]
        for filename in os.listdir('.'):
            if os.path.isfile(filename):
                for pattern in patterns:
                    if pattern.match(filename):
                        os.remove(filename)

        self.structure.set_constraint(self.constraints + self.hookean_constraints)
        if optimizer == 'BFGS':
            amh = AdjustedMinimaHopping(atoms=self.structure, optimizer=BFGS)
        elif optimizer == 'FIRE':
            amh = AdjustedMinimaHopping(atoms=self.structure, optimizer=FIRE2)
        amh(totalsteps=totalsteps, max_optimize_steps=max_optimize_steps)
        minima_traj = Trajectory('minima.traj')
        minima_list = []
        for idx, atoms in enumerate(minima_traj):
            minima_now = atoms.get_potential_energy()
            minima_list.append([atoms, minima_now])
        minima_list = sorted(minima_list, key=lambda x: x[1])
        with open('MINIMA_EE', 'w') as file:
            for idx, minima in enumerate(minima_list):
                write_vasp(f'MINIMA{idx}.vasp', minima[0])
                file.write(f'{minima[1]}\n')


if __name__ == "__main__":

    checkpoint_path = 'checkpoint.pt'

    fj = FairchemJob.from_checkpoint(
        checkpoint_path,
        overwrite={'trainer': 's2ef',
                   'dataset': {'lin_ref': None, 'oc20_ref': None},
                   'val_dataset': {'lin_ref': None, 'oc20_ref': None},
                   'optim': {'load_balancing': None, 'eval_every': None, 'max_epochs': 1000}})
    fj.load_datasets(train_set_path='train', valid_set_path='valid')
    # fj.train(best_epoch_starter=30, best_epoch_criteria=20)
    # predictions = fj.evaluate()
    # print(predictions)
    # print(np.load(fj.result_path))
