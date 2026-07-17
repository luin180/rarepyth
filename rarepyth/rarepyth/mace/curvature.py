import os
import glob
import numpy as np
import ase.io
from ase.vibrations import Vibrations
from ase.units import _hbar, _e

_amu = 1.66053906660e-27
_angstrom = 1e-10
_EIGVAL_TO_EV = _hbar / _e * np.sqrt(_e / (_amu * _angstrom ** 2))
_EV_TO_CM = 8065.544


class CurvatureSampler:
    def __init__(self, model_path, device='cuda', default_dtype='float64', enable_cueq=True):
        from mace.calculators import MACECalculator

        self.model_path = model_path
        self.calc = MACECalculator(
            model_paths=model_path if isinstance(model_path, list) else [model_path],
            device=device,
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
        )
        self.atoms = None
        self.free_indices = None
        self._hessian = None
        self._frequencies_ev = None
        self._frequencies_cm = None
        self._eigenvectors = None
        self._eigenvalues = None

    def load_structure(self, poscar_path='POSCAR.vasp'):
        self.atoms = ase.io.read(poscar_path)
        self.atoms.calc = self.calc

    def set_free_indices(self, tag_file='TAG'):
        if os.path.exists(tag_file):
            free = []
            with open(tag_file, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) >= 2 and int(parts[1]) != 0:
                        free.append(i)
            self.free_indices = free
            return free

        if self.atoms.constraints:
            from ase.constraints import FixAtoms
            fixed = set()
            for c in self.atoms.constraints:
                if isinstance(c, FixAtoms):
                    fixed.update(c.get_indices())
            free = [i for i in range(len(self.atoms)) if i not in fixed]
            self.free_indices = free
            return free

        self.free_indices = list(range(len(self.atoms)))
        return self.free_indices

    def _clean_vib_cache(self):
        for f in glob.glob('vib.*'):
            try:
                os.remove(f)
            except OSError:
                pass

    def analyze_modes(self, delta=0.01):
        if self.atoms is None:
            raise RuntimeError('No structure loaded. Call load_structure() first.')
        if self.free_indices is None:
            self.set_free_indices()

        self._clean_vib_cache()

        vib = Vibrations(
            atoms=self.atoms,
            indices=self.free_indices,
            name='vib',
            delta=delta,
        )
        vib.run()
        raw_energies_ev = vib.get_energies()

        H_raw = vib.H.copy()
        vib.clean()

        masses = self.atoms.get_masses()[self.free_indices]
        n_free = len(self.free_indices)
        mass_sqrt_inv = 1.0 / np.sqrt(np.repeat(masses, 3))
        M_inv_sqrt = np.diag(mass_sqrt_inv)
        D = M_inv_sqrt @ H_raw @ M_inv_sqrt
        D = (D + D.T) / 2.0

        eigenvalues, eigenvectors = np.linalg.eigh(D)

        freqs_ev = np.zeros(n_free * 3)
        for i in range(n_free * 3):
            if eigenvalues[i] > 1e-12:
                freqs_ev[i] = _EIGVAL_TO_EV * np.sqrt(eigenvalues[i])
            elif eigenvalues[i] < -1e-12:
                freqs_ev[i] = -_EIGVAL_TO_EV * np.sqrt(abs(eigenvalues[i]))
            else:
                freqs_ev[i] = 0.0

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._frequencies_ev = freqs_ev
        self._frequencies_cm = freqs_ev * _EV_TO_CM

        return freqs_ev, eigenvectors, masses

    def select_soft_modes(self, freq_threshold_cm=50.0):
        if self._frequencies_cm is None:
            raise RuntimeError('Call analyze_modes() first.')

        freqs = self._frequencies_cm
        soft = []
        for i, f in enumerate(freqs):
            if 0.0 < f < freq_threshold_cm:
                soft.append(i)

        n_imag = sum(1 for f in freqs if f < 0.0)
        if n_imag > 0:
            soft = list(range(n_imag)) + soft

        return soft

    def generate_displacements(self, delta=0.01, freq_threshold_cm=50.0):
        if self._eigenvectors is None:
            raise RuntimeError('Call analyze_modes() first.')

        soft_modes = self.select_soft_modes(freq_threshold_cm=freq_threshold_cm)
        if not soft_modes:
            return []

        masses = self.atoms.get_masses()[self.free_indices]
        n_free = len(self.free_indices)

        displaced = []
        for mode_idx in soft_modes:
            v_mass = self._eigenvectors[:, mode_idx]
            v_cart = v_mass / np.sqrt(np.repeat(masses, 3))
            norm = np.linalg.norm(v_cart)
            if norm < 1e-14:
                continue
            v_cart /= norm
            v_cart *= delta

            v_3d = v_cart.reshape(n_free, 3)

            for sign, label in [(+1, '+'), (-1, '-')]:
                atoms_disp = self.atoms.copy()
                for j, atom_idx in enumerate(self.free_indices):
                    atoms_disp.positions[atom_idx] += sign * v_3d[j]
                displaced.append({
                    'mode_index': mode_idx,
                    'frequency_ev': self._frequencies_ev[mode_idx],
                    'frequency_cm': self._frequencies_cm[mode_idx],
                    'sign': label,
                    'atoms': atoms_disp,
                    'displacement_vector': sign * v_cart,
                })

        return displaced

    def export_for_vasp(self, output_dir='.', delta=0.01, freq_threshold_cm=50.0):
        displaced = self.generate_displacements(delta=delta, freq_threshold_cm=freq_threshold_cm)
        os.makedirs(output_dir, exist_ok=True)

        paths = []
        for entry in displaced:
            fname = f'POSCAR_DISP_M{entry["mode_index"]}_{entry["sign"]}.vasp'
            fpath = os.path.join(output_dir, fname)
            ase.io.write(fpath, entry['atoms'], format='vasp', direct=True, sort=False, vasp5=True)
            paths.append((fpath, entry))

        eq_path = os.path.join(output_dir, 'POSCAR_EQ.vasp')
        ase.io.write(eq_path, self.atoms, format='vasp', direct=True, sort=False, vasp5=True)

        return paths, eq_path
