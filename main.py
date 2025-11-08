"""
main.py
Mesh Normalization, Quantization, Reconstruction, and Error Analysis

Place .obj meshes in ./meshes/ and run:
    python main.py

Outputs go into ./output/
"""

import os
import json
import math
import random
from glob import glob
from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# --------------------------
# Config
# --------------------------
MESH_FOLDER = "meshes"
OUT_FOLDER = "output"
N_BINS = 1024
RANDOM_ROTATIONS = 5   # number of random rotations for invariance tests
ADAPTIVE_NEIGHBORS = 20  # k for local density estimation
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# helper utilities
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

# --------------------------
# IO helpers
# --------------------------
def load_mesh(path):
    mesh = trimesh.load(path, process=False)
    if not hasattr(mesh, 'vertices'):
        raise ValueError(f"Failed to load vertices from {path}")
    return mesh

def save_mesh_vertices_as_obj(vertices, faces, path):
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    m.export(path)

# --------------------------
# Stats & printing
# --------------------------
def vertex_stats(vertices):
    stats = {}
    stats['count'] = int(vertices.shape[0])
    stats['min'] = vertices.min(axis=0).tolist()
    stats['max'] = vertices.max(axis=0).tolist()
    stats['mean'] = vertices.mean(axis=0).tolist()
    stats['std'] = vertices.std(axis=0).tolist()
    return stats

def print_stats(name, stats):
    print(f"--- {name} ---")
    print(f"Vertex count: {stats['count']}")
    for i, axis in enumerate(['x','y','z']):
        print(f"{axis}: min={stats['min'][i]:.6f}, max={stats['max'][i]:.6f}, mean={stats['mean'][i]:.6f}, std={stats['std'][i]:.6f}")
    print()

# --------------------------
# Normalization methods
# --------------------------
def minmax_normalize(vertices):
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    denom = v_max - v_min
    denom = np.where(denom == 0, 1.0, denom)
    normalized = (vertices - v_min) / denom
    meta = {'type': 'minmax', 'v_min': v_min.tolist(), 'v_max': v_max.tolist()}
    return normalized, meta

def minmax_denormalize(normalized, meta):
    v_min = np.array(meta['v_min'])
    v_max = np.array(meta['v_max'])
    return normalized * (v_max - v_min) + v_min

def unit_sphere_normalize(vertices):
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    dists = np.linalg.norm(centered, axis=1)
    maxd = dists.max() if dists.max() > 0 else 1.0
    normalized = centered / maxd
    meta = {'type': 'unit_sphere', 'centroid': centroid.tolist(), 'scale': float(maxd)}
    return normalized, meta

def unit_sphere_denormalize(normalized, meta):
    centroid = np.array(meta['centroid'])
    scale = float(meta['scale'])
    return normalized * scale + centroid

# --------------------------
# Quantization
# --------------------------
def quantize(normalized, n_bins=N_BINS):
    if normalized.min() < 0 or normalized.max() > 1:
        normalized_mapped = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-12)
    else:
        normalized_mapped = normalized.copy()
    q = np.floor(normalized_mapped * (n_bins - 1)).astype(int)
    q = np.clip(q, 0, n_bins - 1)
    return q, normalized_mapped

def dequantize(q, n_bins=N_BINS):
    return q.astype(float) / float(n_bins - 1)

# --------------------------
# Error metrics
# --------------------------
def compute_errors(orig, recon):
    diff = orig - recon
    mse_per_axis = np.mean(diff**2, axis=0)
    mae_per_axis = np.mean(np.abs(diff), axis=0)
    mse = float(np.mean(mse_per_axis))
    mae = float(np.mean(mae_per_axis))
    return {'mse_per_axis': mse_per_axis.tolist(), 'mae_per_axis': mae_per_axis.tolist(), 'mse': mse, 'mae': mae}

# --------------------------
# Adaptive quantization helpers (bonus)
# --------------------------
def compute_local_density(vertices, k=ADAPTIVE_NEIGHBORS):
    k_use = min(k, max(2, len(vertices)-1))
    nbrs = NearestNeighbors(n_neighbors=k_use).fit(vertices)
    distances, _ = nbrs.kneighbors(vertices)
    mean_dist = distances[:, 1:].mean(axis=1)
    return mean_dist

def adaptive_bin_assignment(vertices_norm_mapped, base_bins=N_BINS, k=ADAPTIVE_NEIGHBORS):
    local_density = compute_local_density(vertices_norm_mapped, k=k)
    ld_min, ld_max = local_density.min(), local_density.max()
    if ld_max - ld_min < 1e-12:
        norm_ld = np.zeros_like(local_density)
    else:
        norm_ld = (local_density - ld_min) / (ld_max - ld_min)
    buckets = np.digitize(norm_ld, bins=[0.33, 0.66])
    bin_map = {0: int(base_bins * 1.2), 1: int(base_bins * 1.0), 2: int(base_bins * 0.7)}
    per_vertex_bins = np.array([bin_map[b] for b in buckets])
    return per_vertex_bins, buckets

def quantize_adaptive(normalized_mapped, base_bins=N_BINS, k=ADAPTIVE_NEIGHBORS):
    per_vertex_bins, buckets = adaptive_bin_assignment(normalized_mapped, base_bins, k=k)
    q = np.zeros_like(normalized_mapped, dtype=int)
    for i in range(normalized_mapped.shape[0]):
        bins = int(per_vertex_bins[i])
        q[i, :] = np.floor(normalized_mapped[i, :] * (bins - 1)).astype(int)
    return q, per_vertex_bins, buckets

def dequantize_adaptive(q, per_vertex_bins):
    deq = np.zeros_like(q, dtype=float)
    for i in range(q.shape[0]):
        bins = int(per_vertex_bins[i])
        deq[i, :] = q[i, :].astype(float) / float(bins - 1)
    return deq

# --------------------------
# Plot helpers
# --------------------------
def plot_error_axes(mse_per_axis, mae_per_axis, out_path_prefix):
    axes = ['x','y','z']
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(axes, mse_per_axis)
    ax.set_title("MSE per axis")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(out_path_prefix + "_mse_axis.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(axes, mae_per_axis)
    ax.set_title("MAE per axis")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(out_path_prefix + "_mae_axis.png")
    plt.close(fig)

def plot_error_hist(errors, out_path):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(errors, bins=100)
    ax.set_title("Reconstruction error distribution (L2 per-vertex)")
    ax.set_xlabel("L2 error")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# --------------------------
# Main flow per mesh
# --------------------------
def process_mesh_file(mesh_path, out_root):
    mesh_name = Path(mesh_path).stem
    print(f"Processing {mesh_name} ...")
    mesh = load_mesh(mesh_path)
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy() if mesh.faces is not None else None

    out_stats = os.path.join(out_root, "stats")
    out_norm = os.path.join(out_root, "normalized")
    out_quant = os.path.join(out_root, "quantized")
    out_recon = os.path.join(out_root, "reconstructed")
    out_plots = os.path.join(out_root, "plots")
    for d in [out_stats, out_norm, out_quant, out_recon, out_plots]:
        ensure_dir(d)

    stats = vertex_stats(verts)
    print_stats(mesh_name, stats)
    with open(os.path.join(out_stats, f"{mesh_name}_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    results = {}

    norm_methods = [
        ('minmax', minmax_normalize, minmax_denormalize),
        ('unit_sphere', unit_sphere_normalize, unit_sphere_denormalize)
    ]
    for method_name, normalize_fn, denorm_fn in norm_methods:
        normalized, meta = normalize_fn(verts)
        norm_path = os.path.join(out_norm, f"{mesh_name}_{method_name}_normalized.obj")
        save_mesh_vertices_as_obj(normalized, faces, norm_path)

        q, normalized_mapped = quantize(normalized, n_bins=N_BINS)
        deq_simple = dequantize(q, n_bins=N_BINS)

        if normalized.min() < 0 or normalized.max() > 1:
            nm_min, nm_max = normalized.min(), normalized.max()
            deq_mapped_back = deq_simple * (nm_max - nm_min) + nm_min
        else:
            deq_mapped_back = deq_simple

        quant_path = os.path.join(out_quant, f"{mesh_name}_{method_name}_quantized.obj")
        save_mesh_vertices_as_obj(deq_mapped_back, faces, quant_path)

        deq = dequantize(q, n_bins=N_BINS)
        if normalized.min() < 0 or normalized.max() > 1:
            nm_min, nm_max = normalized.min(), normalized.max()
            deq_normalized = deq * (nm_max - nm_min) + nm_min
        else:
            deq_normalized = deq

        reconstructed = denorm_fn(deq_normalized, meta)
        recon_path = os.path.join(out_recon, f"{mesh_name}_{method_name}_reconstructed.obj")
        save_mesh_vertices_as_obj(reconstructed, faces, recon_path)

        errs = compute_errors(verts, reconstructed)
        results[f"{method_name}"] = errs

        out_prefix = os.path.join(out_plots, f"{mesh_name}_{method_name}")
        plot_error_axes(errs['mse_per_axis'], errs['mae_per_axis'], out_prefix)
        l2_errors = np.linalg.norm(verts - reconstructed, axis=1)
        plot_error_hist(l2_errors, out_prefix + "_l2hist.png")

    rot_results = {'uniform': [], 'adaptive': []}
    for r in range(RANDOM_ROTATIONS):
        axis = np.random.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2*np.pi)
        ux, uy, uz = axis
        c = np.cos(angle); s = np.sin(angle); C = 1 - c
        R = np.array([
            [ux*ux*C + c,    ux*uy*C - uz*s, ux*uz*C + uy*s],
            [uy*ux*C + uz*s, uy*uy*C + c,    uy*uz*C - ux*s],
            [uz*ux*C - uy*s, uz*uy*C + ux*s, uz*uz*C + c]
        ])
        t = np.random.uniform(-0.5, 0.5, size=3)

        verts_rt = (verts @ R.T) + t

        normalized_rt, meta_rt = minmax_normalize(verts_rt)
        q_rt, norm_rt_mapped = quantize(normalized_rt, n_bins=N_BINS)
        deq_rt = dequantize(q_rt, n_bins=N_BINS)
        if normalized_rt.min() < 0 or normalized_rt.max() > 1:
            nm_min, nm_max = normalized_rt.min(), normalized_rt.max()
            deq_rt_n = deq_rt * (nm_max - nm_min) + nm_min
        else:
            deq_rt_n = deq_rt
        recon_rt = minmax_denormalize(deq_rt_n, meta_rt)
        e_uniform = compute_errors(verts_rt, recon_rt)
        rot_results['uniform'].append(e_uniform['mse'])

        q_adapt, per_vertex_bins, buckets = quantize_adaptive(norm_rt_mapped, base_bins=N_BINS, k=ADAPTIVE_NEIGHBORS)
        deq_adapt = dequantize_adaptive(q_adapt, per_vertex_bins)
        if normalized_rt.min() < 0 or normalized_rt.max() > 1:
            nm_min, nm_max = normalized_rt.min(), normalized_rt.max()
            deq_adapt_n = deq_adapt * (nm_max - nm_min) + nm_min
        else:
            deq_adapt_n = deq_adapt
        recon_adapt = minmax_denormalize(deq_adapt_n, meta_rt)
        e_adapt = compute_errors(verts_rt, recon_adapt)
        rot_results['adaptive'].append(e_adapt['mse'])

    rot_summary = {
        'uniform_mean_mse': float(np.mean(rot_results['uniform'])),
        'uniform_std_mse': float(np.std(rot_results['uniform'])),
        'adaptive_mean_mse': float(np.mean(rot_results['adaptive'])),
        'adaptive_std_mse': float(np.std(rot_results['adaptive'])),
        'uniform_raw': rot_results['uniform'],
        'adaptive_raw': rot_results['adaptive']
    }

    results['rotation_invariance_test'] = rot_summary

    with open(os.path.join(out_root, f"{mesh_name}_results_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Finished {mesh_name}. Results saved to {out_root}\n")
    return results

# --------------------------
# Main entry
# --------------------------
def main():
    ensure_dir(OUT_FOLDER)

    mesh_paths = glob(os.path.join(MESH_FOLDER, "*.obj"))
    if not mesh_paths:
        print(f"No .obj files found in {MESH_FOLDER}. Please place meshes there and re-run.")
        return

    overall_summary = {}
    for path in mesh_paths:
        mesh_name = Path(path).stem
        out_root = os.path.join(OUT_FOLDER, mesh_name)
        ensure_dir(out_root)
        res = process_mesh_file(path, out_root)
        overall_summary[mesh_name] = res

    with open(os.path.join(OUT_FOLDER, "overall_results_summary.json"), "w") as fh:
        json.dump(overall_summary, fh, indent=2)

    print("All done. Check the output/ directory for results, plots and reconstructed meshes.")

if __name__ == "__main__":
    main()
