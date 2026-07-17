# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import ase.io
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from mace.calculators import MACECalculator


class MACEThermo:
    def __init__(self, model_path, structure_path='POSCAR.vasp', device='cuda', default_dtype='float64'):
        """
        Initialize the MACE thermodynamic calculator.
        """
        self.model_path = model_path
        self.structure = ase.io.read(structure_path)

        self.calc = MACECalculator(
            model_paths=model_path,
            device=device,
            default_dtype=default_dtype
        )
        self.structure.calc = self.calc

    def get_vibrating_indices(self, tag_file='TAG'):
        """
        Read the TAG file or structure constraints to identify free atoms.
        """
        indices = []
        if os.path.exists(tag_file):
            with open(tag_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                for i, line in enumerate(lines):
                    if int(line.split()[1]) != 0:
                        indices.append(i)
            print(f"Loaded {len(indices)} vibrating atoms from {tag_file}.")
            return indices

        if self.structure.constraints:
            from ase.constraints import FixAtoms
            fixed = set()
            for constr in self.structure.constraints:
                if isinstance(constr, FixAtoms):
                    fixed.update(constr.get_indices())
            indices = [i for i in range(len(self.structure)) if i not in fixed]
            print(f"Loaded {len(indices)} vibrating atoms from structure constraints.")
            return indices

        print("Warning: No TAG or constraints found. All atoms will be displaced.")
        return list(range(len(self.structure)))

    def run_thermo(self, temperature=298.15, delta=0.01, cutoff_energy=0.0062):
        """
        Run Vibrations (Gamma-point partial Hessian), extract frequencies,
        apply low-frequency clamping, and calculate thermodynamic corrections.
        """
        indices = self.get_vibrating_indices()

        if not indices:
            print("Error: No free atoms found for vibrational analysis.")
            return None

        print(f"Starting Vibrations calculation at {temperature} K...")

        # Forcefully clean all legacy vibration cache files to prevent dimension mismatch
        for f in glob.glob('vib.*'):
            try:
                os.remove(f)
            except OSError:
                pass

        vib = Vibrations(
            atoms=self.structure,
            indices=indices,
            name='vib',
            delta=delta
        )

        # Run finite displacement
        vib.run()
        energies = vib.get_energies()

        real_energies = []
        imaginary_energies = []

        for e in energies:
            # Safely extract real and imaginary parts to avoid NumPy broadcasting traps
            real_part = float(np.real(e))
            imag_part = float(np.imag(e))

            # 1. Pure imaginary frequency (complex format)
            if abs(imag_part) > 1e-4:
                imaginary_energies.append(imag_part)
            # 2. Pure imaginary frequency (negative format in some ASE versions)
            elif real_part < -1e-4:
                imaginary_energies.append(real_part)
            # 3. Real vibrational frequency
            elif real_part > 1e-4:
                if real_part < cutoff_energy:
                    real_energies.append(cutoff_energy)
                else:
                    real_energies.append(real_part)

        # Clean cache immediately after use
        vib.clean()

        if imaginary_energies:
            print(f"Warning: Found {len(imaginary_energies)} imaginary frequencies.")

        if not real_energies:
            print("Error: No real frequencies found. Cannot calculate thermodynamics.")
            return None

        # Harmonic thermodynamics calculation
        thermo = HarmonicThermo(real_energies)

        # Manually calculate ZPE to ensure compatibility across ASE versions
        zpe = sum(real_energies) / 2.0

        results = {
            'temperature': temperature,
            'ZPE': zpe,
            'entropy': thermo.get_entropy(temperature=temperature, verbose=False),
            'internal_energy': thermo.get_internal_energy(temperature=temperature, verbose=False),
            'helmholtz_free_energy': thermo.get_helmholtz_energy(temperature=temperature, verbose=False)
        }

        return results
