'''fe_selector.py -- Free Energy Candidate Selection via Multi-Feature Prioritization

Central module for the mace_al_fe pipeline. Handles:
  1. Parsing MACE_THERMO_RESULTS (including extended header metadata)
  2. Computing candidate features (Groups A–E)
  3. Tier-based selection of candidates for DFT free energy validation
  4. Post-hoc Spearman correlation analysis between features and dZPE

Selection logic:
  Tier 1 (mandatory): n_imag > 0  ->  must validate DFT FE
  Tier 2 (mandatory): |F|_max > f_max_threshold  ->  must validate DFT FE
  Tier 3 (diversity fill): stratified random sampling from remainder,
      stratified by MACE ZPE to ensure coverage of different well depths.

After DFT FE completes and dZPE is known, compute_feature_correlations()
ranks all features by their Spearman rho against dZPE.
'''

import os, math, csv, json
import numpy as np
from collections import defaultdict


# ── Header key names used in MACE_THERMO_RESULTS ──
HEADER_KEYS = {
    'n_imag', 'sigma_imag_abs', 'n_soft_raw', 'freq_min_real_cm',
    'freq_max_cm', 'freq_mean_cm', 'freq_std_cm', 'freq_skew',
    'ZPE_raw_eV', 'ZPE_clamped_eV', 'ZPE_ratio',
    'freq_gap_1_2_cm', 'frac_below_100',
    'hessian_cond', 'hessian_trace', 'hessian_anisotropy', 'log_det_hessian',
    'F_max_mace', 'F_rms_mace', 'F_max_ads',
    'n_free', 'n_ads', 'ratio_free_total',
    'E_mace_eV',
}


def parse_thermo_results(filepath):
    '''Parse MACE_THERMO_RESULTS including #-comment header metadata.

    Returns (header: dict, data_lines: list[dict]) where each data dict
    has keys: temperature, ZPE, entropy, internal_energy, helmholtz_free_energy.
    '''
    header = {}
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#') and '=' in line and not line.startswith('# T(K)'):
                # header metadata line:  # key=value
                content = line.lstrip('#').strip()
                if '=' in content:
                    key, val = content.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    try:
                        header[key] = float(val)
                    except ValueError:
                        header[key] = val
            elif line.startswith('#') or not line[0].isdigit():
                continue
            else:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        data.append({
                            'temperature': float(parts[0]),
                            'ZPE': float(parts[1]),
                            'entropy': float(parts[2]),
                            'internal_energy': float(parts[3]),
                            'helmholtz_free_energy': float(parts[4]),
                        })
                    except ValueError:
                        continue
    return header, data


def _get_tag_groups(poscar_path, tag_file=None):
    '''Read TAG or selective_dynamics to partition atoms.

    Returns (slab_indices, adsorbate_indices).
    '''
    from ase.io import read
    atoms = read(poscar_path)

    if tag_file and os.path.exists(tag_file):
        slab_idx = []
        ads_idx = []
        with open(tag_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 2:
                    tag = int(parts[1])
                    if tag == 0:
                        slab_idx.append(i)
                    else:
                        ads_idx.append(i)
        if slab_idx or ads_idx:
            return slab_idx, ads_idx

    # fallback: selective_dynamics + height
    slab_idx = []
    ads_idx = []
    z_coords = atoms.positions[:, 2]
    z_max = z_coords.max()
    threshold = z_max - 3.0  # Ang
    for i in range(len(atoms)):
        if z_coords[i] < threshold:
            slab_idx.append(i)
        else:
            ads_idx.append(i)
    return slab_idx, ads_idx


def extract_geometry_features(poscar_path, tag_file='TAG'):
    '''Extract E-group geometry features from a POSCAR.

    Returns dict with keys: z_ads_com, min_dist_ads_surf, n_atoms_total,
    n_free, n_ads, cell_area, ads_mol_elongation.
    '''
    import numpy as np
    from ase.io import read

    atoms = read(poscar_path)
    slab_idx, ads_idx = _get_tag_groups(poscar_path, tag_file)

    feats = {}
    feats['n_atoms_total'] = len(atoms)
    feats['n_free'] = len(ads_idx)
    feats['n_ads'] = len(ads_idx)

    cell = atoms.cell
    feats['cell_area'] = np.linalg.norm(np.cross(cell[0], cell[1]))

    if not ads_idx:
        feats['z_ads_com'] = 0.0
        feats['min_dist_ads_surf'] = 0.0
        feats['ads_mol_elongation'] = 1.0
        feats['ratio_free_total'] = 0.0
        return feats

    feats['ratio_free_total'] = len(ads_idx) / len(atoms) if len(atoms) > 0 else 0.0

    ads_positions = atoms.positions[ads_idx]
    feats['z_ads_com'] = float(np.mean(ads_positions[:, 2]))

    if slab_idx:
        slab_top_z = atoms.positions[slab_idx, 2].max()
        feats['z_ads_com_rel'] = feats['z_ads_com'] - slab_top_z
        slab_positions = atoms.positions[slab_idx]
        min_dists = []
        for a_pos in ads_positions:
            diffs = slab_positions - a_pos
            disp_vecs = diffs @ np.linalg.inv(cell).T
            disp_vecs -= np.round(disp_vecs)
            dists = np.linalg.norm(disp_vecs @ cell.T, axis=1)
            min_dists.append(dists.min())
        feats['min_dist_ads_surf'] = float(min(min_dists)) if min_dists else 0.0
    else:
        feats['z_ads_com_rel'] = 0.0
        feats['min_dist_ads_surf'] = 0.0

    if len(ads_idx) >= 2:
        ads_centered = ads_positions - ads_positions.mean(axis=0)
        cov = ads_centered.T @ ads_centered / (len(ads_idx) - 1)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_sorted = np.sort(eigvals[eigvals > 1e-12])
        if len(eigvals_sorted) >= 2:
            feats['ads_mol_elongation'] = float(np.sqrt(eigvals_sorted[-1] / eigvals_sorted[0]))
        else:
            feats['ads_mol_elongation'] = 1.0
    else:
        feats['ads_mol_elongation'] = 1.0

    for key in ['z_ads_com', 'min_dist_ads_surf', 'cell_area', 'z_ads_com_rel']:
        if key in feats:
            feats[key] = float(feats[key])

    return feats


def build_candidate_record(thermo_header, thermo_data, geo_feats, 
                            system_name, dft_sp_energy=None, mace_global_min=None):
    '''Merge all feature groups into a single candidate dict.

    Args:
        thermo_header: dict from parse_thermo_results header
        thermo_data: list[dict] from parse_thermo_results data rows
        geo_feats: dict from extract_geometry_features
        system_name: str
        dft_sp_energy: float or None -- DFT single-point energy (D-group)
        mace_global_min: float or None -- global minimum MACE energy (D-group)

    Returns dict with all features + metadata.
    '''
    rec = {'system': system_name}

    # A/B/C group features from MACE thermo header
    for k in HEADER_KEYS:
        rec[k] = thermo_header.get(k, None)

    # Thermo data at T=298.15
    t298 = None
    for row in thermo_data:
        if abs(row['temperature'] - 298.15) < 1.0:
            t298 = row
            break
    if t298 is None and thermo_data:
        t298 = thermo_data[0]

    if t298:
        rec['ZPE_MACE_eV'] = t298['ZPE']
        rec['entropy_MACE'] = t298['entropy']
        rec['helmholtz_MACE_eV'] = t298['helmholtz_free_energy']
        rec['internal_energy_MACE_eV'] = t298['internal_energy']
    else:
        rec['ZPE_MACE_eV'] = None
        rec['entropy_MACE'] = None
        rec['helmholtz_MACE_eV'] = None
        rec['internal_energy_MACE_eV'] = None

    # D-group features
    if dft_sp_energy is not None and mace_global_min is not None:
        rec['E_mace_rel'] = (thermo_header.get('E_mace_eV', 0) or 0) - mace_global_min
        rec['E_dft_rel'] = dft_sp_energy - mace_global_min  # approximate
        rec['dE_SP'] = abs((thermo_header.get('E_mace_eV', 0) or 0) - dft_sp_energy)
    else:
        rec['E_mace_rel'] = None
        rec['E_dft_rel'] = None
        rec['dE_SP'] = None

    # E-group features
    for k, v in geo_feats.items():
        if k not in rec:
            rec[k] = v

    return rec


def select_candidates(candidates, n_select=20, f_max_threshold=0.05):
    '''Tier-based selection.

    Args:
        candidates: list[dict] from build_candidate_record
        n_select: int, maximum total candidates to select
        f_max_threshold: float (eV/Ang), Tier-2 threshold for residual force

    Returns:
        selected: list[dict] in priority order
        tiers: dict with 'tier1', 'tier2', 'tier3' lists
    '''
    tier1 = []
    tier2 = []
    tier3 = []

    for c in candidates:
        n_imag = c.get('n_imag', 0) or 0
        f_max = c.get('F_max_mace') or 0.0
        if n_imag > 0:
            tier1.append(c)
        elif f_max > f_max_threshold:
            tier2.append(c)
        else:
            tier3.append(c)

    # Tier-3: stratified by ZPE_MACE for diversity
    tier3_zpe = [(c.get('ZPE_MACE_eV') or 0.0, c) for c in tier3]
    tier3_zpe.sort(key=lambda x: x[0])

    n_strata = 4
    stratum_size = max(1, len(tier3_zpe) // n_strata)
    strata = []
    for s in range(n_strata):
        sl = tier3_zpe[s * stratum_size : (s + 1) * stratum_size] if s < n_strata - 1 else tier3_zpe[s * stratum_size:]
        strata.append(sl)

    selected = list(tier1) + list(tier2)
    remaining = n_select - len(selected)
    if remaining <= 0:
        return selected[:n_select], {'tier1': tier1, 'tier2': tier2, 'tier3': []}

    tier3_selected = []
    for s in strata:
        if len(tier3_selected) >= remaining:
            break
        if not s:
            continue
        n_from_stratum = max(1, remaining // len(strata))
        n_from_stratum = min(n_from_stratum, len(s))
        picked = [x[1] for x in s[:n_from_stratum]]
        tier3_selected.extend(picked)
    tier3_selected = tier3_selected[:remaining]

    selected.extend(tier3_selected)
    return selected, {'tier1': tier1, 'tier2': tier2, 'tier3': tier3_selected}


def compute_feature_correlations(comparison_records, target_key='dZPE_eV'):
    '''Compute Spearman ρ between candidate features and a target metric.

    Args:
        comparison_records: list[dict], each must have the target_key field
            and the candidate features from build_candidate_record.
        target_key: str, the field name to correlate against (default 'dZPE_eV').

    Returns:
        correlations: list of (feature_name, rho, p_value) sorted by |rho| descending.
    '''
    if len(comparison_records) < 4:
        return []

    from scipy.stats import spearmanr

    # Collect all feature names (numeric only)
    feature_names = []
    for rec in comparison_records:
        for k, v in rec.items():
            if k in ('system', 'tiers', 'data_line'):
                continue
            if isinstance(v, (int, float)) and v is not None:
                if k not in feature_names:
                    feature_names.append(k)

    correlations = []
    for feat in feature_names:
        x_vals = []
        y_vals = []
        for rec in comparison_records:
            xv = rec.get(feat)
            yv = rec.get(target_key)
            if xv is not None and yv is not None and not (isinstance(xv, float) and np.isnan(xv)):
                x_vals.append(float(xv))
                y_vals.append(float(yv))
        if len(x_vals) < 4:
            continue
        try:
            rho, p = spearmanr(x_vals, y_vals)
            if not np.isnan(rho):
                correlations.append((feat, rho, p))
        except Exception:
            pass

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    return correlations


def write_feature_csv(candidates, filepath, extra_fields=None):
    '''Write all candidate features to CSV for downstream analysis.

    Args:
        candidates: list[dict] from build_candidate_record
        filepath: str
        extra_fields: list[str] -- additional field names to include
    '''
    fieldnames = ['system']
    fieldnames.extend(sorted(HEADER_KEYS))
    fieldnames.extend(['ZPE_MACE_eV', 'entropy_MACE', 'helmholtz_MACE_eV'])
    fieldnames.extend(['E_mace_rel', 'E_dft_rel', 'dE_SP'])
    fieldnames.extend(['z_ads_com', 'z_ads_com_rel', 'min_dist_ads_surf',
                       'n_atoms_total', 'n_free', 'n_ads', 'cell_area',
                       'ads_mol_elongation', 'ratio_free_total'])
    if extra_fields:
        for f in extra_fields:
            if f not in fieldnames:
                fieldnames.append(f)

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for c in candidates:
            row = {k: c.get(k) for k in fieldnames}
            writer.writerow(row)

    print(f'  Features CSV written: {filepath} ({len(candidates)} rows)')
