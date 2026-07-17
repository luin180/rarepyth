# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:48:57 2024

@author: Wang Junhao

Ref: Surface Science, 2013, 617, 53-59, doi:10.1016/j.susc.2013.05.016.
"""

import math
from math import sqrt
from functools import cached_property

import numpy as np
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.surface import SlabGenerator, Slab, center_slab

from rarepyth.base import _mswindows, Message
from rarepyth.utils import fix_atoms_by_height
from rarepyth.vasp.compute import VaspJob
import rarepyth.vasp.errorhandle as eh


class SlabGeneratorFracional(SlabGenerator):
    def get_slab(
        self,
        shift=0,
        tol=0.1,
        energy=None,
    ):
        """Variant method of pymatgen SlabGenerator class. Fractional cut of
        slabs is supported.
        """
        # Calculate total number of layers
        height = self._proj_height
        height_per_layer = round(height / self.parent.lattice.d_hkl(self.miller_index), 8)

        if self.in_unit_planes:
            min_n_layers_slab = self.min_slab_size / height_per_layer
            n_layers_vac = math.ceil(self.min_vac_size / height_per_layer)
        else:
            min_n_layers_slab = self.min_slab_size / height
            n_layers_vac = math.ceil(self.min_vac_size / height)

        # Get reduced formula
        formula_crystal = {}
        for site in self.oriented_unit_cell:
            if site.species_string in formula_crystal.keys():
                formula_crystal[site.species_string] += 1
            else:
                formula_crystal[site.species_string] = 1
        factor = math.gcd(*formula_crystal.values())
        for key in formula_crystal.keys():
            formula_crystal[key] = formula_crystal[key] // factor

        # Shift all atoms to the termination
        frac_coords = self.oriented_unit_cell.frac_coords
        frac_coords = np.array(frac_coords) + np.array([0, 0, -shift])[None, :]
        frac_coords -= np.floor(frac_coords)  # wrap to the [0, 1) range

        # Find proper z height
        z0 = min_n_layers_slab - int(min_n_layers_slab)
        if math.isclose(z0, 0.0):
            n_layers_slab = min_n_layers_slab
        else:
            for dz in range(int(100 * z0), 101):
                formula = {}
                for i, site in enumerate(self.oriented_unit_cell):
                    if frac_coords[i, 2] <= 0.01 * dz:
                        if site.species_string in formula.keys():
                            formula[site.species_string] += 1
                        else:
                            formula[site.species_string] = 1
                if not len(formula) == len(formula_crystal):
                    continue
                ratio = 0
                for key in formula:
                    if formula[key] % formula_crystal[key]:
                        break
                    else:
                        if not ratio:
                            ratio = formula[key] // formula_crystal[key]
                        else:
                            if not ratio == formula[key] // formula_crystal[key]:
                                break
                else:
                    n_layers_slab = 0.01 * dz + int(min_n_layers_slab)
                    break

        n_layers = n_layers_slab + n_layers_vac

        # Prepare for Slab generation: lattice, species, coords and site_properties
        a, b, c = self.oriented_unit_cell.lattice.matrix
        new_lattice = [a, b, n_layers * c]

        species = self.oriented_unit_cell.species_and_occu

        # Scale down z-coordinate by the number of layers
        frac_coords[:, 2] = frac_coords[:, 2] / n_layers

        # Duplicate atom layers by stacking along the z-axis
        all_coords = []
        all_species = species * int(n_layers_slab)
        for idx in range(int(n_layers_slab)):
            _frac_coords = frac_coords.copy()
            _frac_coords[:, 2] += idx / n_layers
            all_coords.extend(_frac_coords)

        # Scale properties by number of atom layers (excluding vacuum)
        props = self.oriented_unit_cell.site_properties
        props = {k: v * int(n_layers_slab) for k, v in props.items()}

        # Add atoms of decimal part; If there is no decimal part, skip this block to avoid incorrect repetition
        if not dz == 100:
            try:
                for i, site in enumerate(self.oriented_unit_cell):
                    if frac_coords[i, 2] <= 0.01 * dz / n_layers:
                        all_coords.append(
                            frac_coords[i, :] + np.array([0, 0, int(n_layers_slab) / n_layers])
                        )
                        all_species.append(species[i])
                        for k in self.oriented_unit_cell.site_properties.keys():
                            props[k].append(self.oriented_unit_cell.site_properties[k][i])
            except NameError:
                pass

        # Generate Slab
        struct = Structure(new_lattice, all_species, all_coords, site_properties=props)

        # (Optionally) Post-process the Slab
        # Orthogonalize the structure (through LLL lattice basis reduction)
        scale_factor = self.slab_scale_factor
        if self.lll_reduce:
            # Sanitize Slab (LLL reduction + site sorting + map frac_coords)
            lll_slab = struct.copy(sanitize=True)
            struct = lll_slab

            # Apply reduction on the scaling factor
            mapping = lll_slab.lattice.find_mapping(struct.lattice)
            if mapping is None:
                raise RuntimeError("LLL reduction has failed")
            scale_factor = np.dot(mapping[2], scale_factor)

        # Center the slab layer around the vacuum
        if self.center_slab:
            struct = center_slab(struct)

        # Reduce to primitive cell
        if self.primitive:
            prim_slab = struct.get_primitive_structure(tolerance=tol)
            struct = prim_slab

            if energy is not None:
                energy *= prim_slab.volume / struct.volume

        # Reorient the lattice to get the correctly reduced cell
        ouc = self.oriented_unit_cell.copy()
        if self.primitive:
            # Find a reduced OUC
            slab_l = struct.lattice
            ouc = ouc.get_primitive_structure(
                constrain_latt={
                    "a": slab_l.a,
                    "b": slab_l.b,
                    "alpha": slab_l.alpha,
                    "beta": slab_l.beta,
                    "gamma": slab_l.gamma,
                }
            )

            # Ensure lattice a and b are consistent between the OUC and the Slab
            ouc = ouc if (slab_l.a == ouc.lattice.a and slab_l.b == ouc.lattice.b) else self.oriented_unit_cell

        return Slab(
            struct.lattice,
            struct.species_and_occu,
            struct.frac_coords,
            self.miller_index,
            ouc,
            shift,
            scale_factor,
            reorient_lattice=self.reorient_lattice,
            site_properties=struct.site_properties,
            energy=energy,
        )


def adjust_vacuum_layer(slab, vacuum):
    new_matrix = slab.lattice.matrix.copy()
    new_frac_coords = slab.frac_coords.copy()
    height = slab.lattice.volume / (slab.lattice.a * slab.lattice.b
                                    * np.sin(slab.lattice.gamma * np.pi / 180))

    z_coords = []
    for z in slab.frac_coords[:, 2]:
        if not any(np.isclose(z, z_coords, atol=1e-04)):
            z_coords.append(z)
    z_coords = np.sort(z_coords)
    diff = []
    for i, z in enumerate(z_coords):
        if not i:
            diff.append(z - z_coords[-1] + 1.0)
        else:
            diff.append(z - z_coords[i - 1])
    settle_z = z_coords[diff.index(max(diff))]
    assert max(diff) >= 3.0 / height
    new_frac_coords = new_frac_coords - np.array([0.0, 0.0, settle_z])

    if vacuum:
        assert vacuum >= 3.0
        frac_range = 1 - max(diff)
        multi = frac_range + vacuum / height
        new_matrix[2] = multi * new_matrix[2]
        new_frac_coords[:, 2] = new_frac_coords[:, 2] / multi

    else:
        multi = 1.0
    new_frac_coords = new_frac_coords + np.array([0.0, 0.0, 1 / (multi * height)])

    return Structure(Lattice(new_matrix),
                     slab.species,
                     new_frac_coords,
                     site_properties=slab.site_properties).get_sorted_structure()


def normalize_slab(structure):
    matrix = structure.lattice.matrix
    frac_coords = structure.frac_coords
    new_frac_coords = frac_coords.copy()

    a_len = np.linalg.norm(matrix[0])
    b_len = np.linalg.norm(matrix[1])
    c_len = np.linalg.norm(matrix[2])
    cos_alpha = np.inner(matrix[1], matrix[2]) / (b_len * c_len)
    cos_beta = np.inner(matrix[0], matrix[2]) / (a_len * c_len)
    cos_gamma = np.inner(matrix[0], matrix[1]) / (a_len * b_len)

    a1 = a_len
    b1 = b_len * cos_gamma
    b2 = sqrt(b_len ** 2 - b1 ** 2)
    c1 = c_len * cos_beta
    c2 = (b_len * c_len * cos_alpha - b1 * c1) / b2
    c3 = sqrt(c_len ** 2 - c1 ** 2 - c2 ** 2)
    new_matrix = np.array([[a1, 0.0, 0.0],
                           [b1, b2, 0.0],
                           [c1, c2, c3]])

    ca_coff = (c1 * b2 - c2 * b1) / (a1 * b2)
    cb_coff = c2 / b2

    new_matrix[2][0] = 0.0
    new_matrix[2][1] = 0.0

    for atom, coord in enumerate(new_frac_coords):
        na = coord[0] + coord[2] * ca_coff
        na = na - int(na) + int(na < 0)
        nb = coord[1] + coord[2] * cb_coff
        nb = nb - int(nb) + int(nb < 0)
        nc = coord[2]
        new_frac_coords[atom] = np.array([na, nb, nc])

    structure_air = Structure(Lattice(new_matrix),
                              structure.species,
                              new_frac_coords).get_sorted_structure()

    c_trans = structure_air.cart_coords[:, 2].min() - 1.0
    new_cart_coords = structure_air.cart_coords.copy()
    for atom, coord in enumerate(new_cart_coords):
        new_cart_coords[atom] = np.array([coord[0], coord[1], coord[2] - c_trans])
    return Structure(structure_air.lattice,
                     structure_air.species,
                     new_cart_coords,
                     coords_are_cartesian=True)


def generate_miller_slabs(
    structure,
    miller,
    min_slab_size,
    vacuum,
    slab_size_in_unit_planes=False,
    export=False
):
    """Generate slabs with given miller index, thick and vacuum layer.


    Parameters
    ----------
    structure: pymatgen Structure
        Initial input structure. Should use conventional unit cell.
    miller: array-like
        Target miller index. Note that certain miller indices are equivalent
        to other slabs using symmetry operations.
    min_slab_size: float
        In Angstroms or in units of hkl planes.
    vacuum: float
        In Angstroms.
    slab_size_in_unit_planes: bool, optional, default=False
        If true, set min_slab_size in number of hkl planes.
    export: bool, optional, default=False
        If true, export slabs in poscar format.

    Returns
    -------
    List of slabs with given miller index, thick and vacuum layer.

    """
    if not isinstance(miller, tuple):
        miller = tuple(miller)

    max_index = max(miller)
    miller_slabs = []

    gen = SlabGeneratorFracional(structure,
                                 miller,
                                 min_slab_size,
                                 1.0,
                                 max_normal_search=max_index,
                                 in_unit_planes=slab_size_in_unit_planes)
    slabs = gen.get_slabs()

    for i, slab in enumerate(slabs):
        miller_slabs.append(adjust_vacuum_layer(normalize_slab(slab), vacuum))
        if export:
            miller_slabs[-1].to(f"SLAB{i}{miller}.vasp", fmt="poscar")

    if miller_slabs:
        Message("Find {} miller slabs.".format(len(miller_slabs))).printm()
    else:
        Message("Invalid miller index, please check symmetry.").exitm()

    return miller_slabs


if not _mswindows:

    class ConvergedSlabBuilder:

        def __init__(
            self,
            incar,
            bulk_structure,
            miller,
            KMesh_resolved_value=0.04,
            **calc_details
        ):
            self.incar = incar.copy()
            self.bulk_structure = bulk_structure.copy()
            self.miller = miller if isinstance(miller, tuple) else tuple(miller)
            self.KMesh_resolved_value = KMesh_resolved_value
            self.calc_details = calc_details
            self.converge_criteria = 0.001
            self.generator = SlabGeneratorFracional(self.bulk_structure,
                                                    self.miller,
                                                    2.0,
                                                    15.0,
                                                    primitive=False)
            self.oriented_unit_cell = self.generator.oriented_unit_cell

        def _gene_ouc_kpoints(self):
            vj = VaspJob(None, self.oriented_unit_cell, None, None, None, None, None, None)
            vj.gene_kpoints(self.KMesh_resolved_value, scheme='gamma')
            self.kpoints_ouc = vj.kpoints
            self.kpoints_slab = vj.kpoints.copy()
            self.kpoints_slab.subdivisions[2] = 1

        def _get_energy(self, struct, kpoints=None):
            vj = VaspJob(self.incar, struct, kpoints, None, None, None, None, None)
            if not kpoints:
                vj.gene_kpoints(self.KMesh_resolved_value)
            vj.set_calc_details(**self.calc_details)
            try:
                if self.incar['LDAU']:
                    vj.gene_potcar(option='f')
                else:
                    vj.gene_potcar(option='r')
            except KeyError:
                vj.gene_potcar(option='r')
            vj.try_scf(print_level=1)
            return vj.oszicar.ionic_energies[-1]

        @cached_property
        def bulk_energy_per_atom(self):
            Message("Calculating energy of oriented unit cell").printm()
            self._gene_ouc_kpoints()
            bepa = self._get_energy(self.oriented_unit_cell,
                                    kpoints=self.kpoints_ouc) / len(self.oriented_unit_cell)
            Message("Bulk energy per atom: {} eV".format(bepa)).printm()
            return bepa

        def static_converge_test(
            self,
            start_slab_size=2.0,
            step_size=1.0,
            max_loop=8
        ):
            self.bulk_energy_per_atom
            self.static_surface_energy_list = []
            self.static_slab_list = []
            self.generator.max_normal_search = None
            self.generator.min_slab_size = start_slab_size - 1.0
            for loop in range(max_loop):
                # Make sure new slab have more layers
                while True:
                    self.generator.min_slab_size += step_size
                    slab_tmp = self.generator.get_slabs(ftol=1.0)[0]
                    if (not self.static_slab_list or
                            (len(slab_tmp) > len(self.static_slab_list[-1]))):
                        self.static_slab_list.append(slab_tmp)
                        break

                # Calculate surface energy of current slab
                Message("Calculating energy of {}th slab model".format(loop)).printm()
                slab = self.static_slab_list[-1]
                slab.to(f'SLAB_THICKNESS{self.generator.min_slab_size}.vasp', fmt='poscar')
                self.static_surface_energy_list.append(
                    (self._get_energy(slab, kpoints=self.kpoints_slab)
                     - len(slab) * self.bulk_energy_per_atom)
                    / (2 * slab.surface_area)
                )
                Message("Surface energy of SLAB{}: {} eV/Å²"
                        .format(loop, format(self.static_surface_energy_list[-1], '.6f'))).printm()

                if loop > 0 and abs(
                    self.static_surface_energy_list[-1]
                        - self.static_surface_energy_list[-2]
                ) <= self.converge_criteria:
                    self.static_converge_height = self.generator.min_slab_size
                    Message("Minimum slab size is {} Å".format(self.static_converge_height)).printm()
                    return
            else:
                raise eh.StaticConvergeFailedError

        def get_static_converged_slabs(
            self,
            min_slab_size=None,
            vacuum=15.0
        ):
            if not min_slab_size:
                try:
                    self.static_converge_height
                except AttributeError:
                    self.static_converge_test()
                finally:
                    self.generator.min_slab_size = self.static_converge_height
            else:
                self.generator.min_slab_size = min_slab_size
            self.generator.max_normal_search = max(self.miller)
            static_slabs_dict = {}
            for slab in self.generator.get_slabs():
                adjusted_slab = adjust_vacuum_layer(normalize_slab(slab), vacuum)
                static_slabs_dict[self._get_energy(adjusted_slab)] = adjusted_slab

            sorted_static_slabs = []
            for energy in static_slabs_dict.keys():
                sorted_static_slabs.append(static_slabs_dict[energy])
            return sorted_static_slabs

        def relaxed_converge_test(
            self,
            min_slab_size=None,
            vacuum=15.0,
            step_size=1.0,
            max_loop=8
        ):
            if not min_slab_size:
                try:
                    self.static_converge_height
                except AttributeError:
                    self.static_converge_test()
                finally:
                    self.generator.min_slab_size = self.static_converge_height - step_size
            else:
                self.generator.min_slab_size = min_slab_size - step_size
            self.generator.max_normal_search = max(self.miller)

            for loop in range(max_loop):
                # Make sure new slab have more layers
                while True:
                    self.generator.min_slab_size += step_size
                    slab_tmp = self.generator.get_slabs()[0]
                    try:
                        len_slab
                    except NameError:
                        len_slab = len(slab_tmp)
                        break
                    else:
                        if len(slab_tmp) > len_slab:
                            break

                # Find most stable termination of given min_slab_size
                static_slabs = {}
                for idx, slab in enumerate(self.generator.get_slabs()):
                    adjusted_slab = adjust_vacuum_layer(normalize_slab(slab), vacuum)
                    adjusted_slab.to(f'SLAB_TERMINAL{idx}.vasp', fmt='poscar')
                    surface_area = np.linalg.norm(np.cross(adjusted_slab.lattice.matrix[0],
                                                           adjusted_slab.lattice.matrix[1]))
                    slab_energy = (self._get_energy(adjusted_slab) - len(adjusted_slab) *
                                   self.bulk_energy_per_atom) / (2 * surface_area)
                    Message(f"Surface energy of SLAB_TERMINAL{idx}: {format(slab_energy, '.6f')} eV").printm()
                    static_slabs[slab_energy] = adjusted_slab
                energy_most_stable = min(static_slabs.keys())
                slab_most_stable = static_slabs[energy_most_stable].copy()
                slab_most_stable.to('SLAB_TERMINAL_BEST.vasp', fmt='poscar')
                surface_area = np.linalg.norm(np.cross(slab_most_stable.lattice.matrix[0],
                                                       slab_most_stable.lattice.matrix[1]))
                self.relaxed_surface_energy_list = [energy_most_stable]
                Message("Surface energy of unrelaxed slab : {} eV/Å²"
                        .format(format(self.relaxed_surface_energy_list[0], '.6f'))).printm()

                # Initialize VaspJob for current slab
                vj = VaspJob(self.incar, slab_most_stable, None, None, None, None, None, None)
                vj.incar['ISIF'] = 2
                vj.set_calc_details(**self.calc_details)
                vj.gene_kpoints(self.KMesh_resolved_value)
                try:
                    if self.incar['LDAU']:
                        vj.gene_potcar(option='f')
                    else:
                        vj.gene_potcar(option='r')
                except KeyError:
                    vj.gene_potcar(option='r')

                # Find converged fix height of current slab
                z_array = vj.poscar.cart_coords[:, 2]
                z_max = max(z_array) - 1.0
                while z_max >= max(z_array) / 2:
                    fix_atoms_by_height(
                        vj.poscar,
                        0.0,
                        z_max,
                        coords_are_cartesian=True)
                    vj.try_relax()

                    self.relaxed_surface_energy_list.append(
                        (vj.oszicar.ionic_energies[-1] - len(slab_most_stable) *
                         self.bulk_energy_per_atom) / (2 * surface_area))
                    Message("Surface energy of relaxed slab with z_max = {}: {} eV/Å²"
                            .format(format(z_max, '.6f'),
                                    format(self.relaxed_surface_energy_list[-1], '.6f'))).printm()

                    if abs(self.relaxed_surface_energy_list[-1] - self.relaxed_surface_energy_list[-2]) <= self.converge_criteria:
                        self.relaxed_converged_slab = vj.poscar.copy()
                        Message('Relaxed slab model have been written into POSCAR_RELAXED.vasp.').printm()
                        return
                    z_max = z_max - 1.0

            else:
                raise eh.RelaxedConvergeFailedError


if __name__ == "__main__":
    from rarepyth.vasp.compute import Incar
    structure = Structure.from_file('structure.vasp')
    miller = (1, 0, 0)
    incar = Incar.from_file('INCAR.ini')
    csb = ConvergedSlabBuilder(incar, structure, miller)
    csb.static_converge_test()
    csb.relaxed_converge_test()
