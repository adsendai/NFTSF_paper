#!/usr/bin/env python3
"""
splice_alanine_circular.py
==========================
Converts alanine-dipeptide-3x250ns-backbone-dihedrals.npz into circular-aware
NF-TSF multi-sim .npy files using sin/cos angle encoding.

Why circular-aware encoding?
- Alanine dihedral angles are periodic variables in radians.
- Raw angles have a discontinuity at the -pi/pi boundary although they
  represent the same physical orientation.
- Encoding each selected angle as (sin(theta), cos(theta)) provides a smooth
  representation for learning.

Output format for each saved feature file matches the scalar NF-TSF multi-sim
layout expected by existing pipelines:

    shape (chunk_size, 1 + num_sims)
    col 0  = frame index (0 .. chunk_size-1)
    col 1+ = trajectory chunks (one column per chunk)

Notes:
- Each long trajectory is sliced into non-overlapping chunks.
- The exact same shuffled chunk ordering and train/test split are applied to
  every selected feature file so features remain aligned chunk-by-chunk.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create circular-aware NF-TSF alanine multi-sim files using sin/cos encoding."
    )
    parser.add_argument("--input", default="alanine-dipeptide-3x250ns-backbone-dihedrals.npz")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument(
        "--dihedral",
        default="phi",
        choices=["phi", "psi", "both"],
        help="Which dihedral(s) to encode: phi, psi, or both",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Frames per chunk (non-overlapping)",
    )
    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.8,
        help="Fraction of chunks used for training (rest used for test)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix", default="alanine")
    return parser.parse_args()


def selected_features(dihedral_mode: str) -> List[str]:
    if dihedral_mode == "phi":
        return ["phi_sin", "phi_cos"]
    if dihedral_mode == "psi":
        return ["psi_sin", "psi_cos"]
    return ["phi_sin", "phi_cos", "psi_sin", "psi_cos"]


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be a positive integer.")
    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train_frac must be between 0 and 1 (exclusive).")


def splice_circular_features(
    npz_path: str, dihedral_mode: str, chunk_size: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Build circular features from alanine dihedrals and return chunked arrays.

    Returns:
        features: dict mapping feature name -> array of shape (num_chunks, chunk_size)
        unit_norm_max_dev: dict mapping source angle (phi/psi) -> max abs deviation
                           of sin(theta)^2 + cos(theta)^2 from 1.
    """
    data = np.load(npz_path)

    feature_chunks: Dict[str, List[np.ndarray]] = {
        "phi_sin": [],
        "phi_cos": [],
        "psi_sin": [],
        "psi_cos": [],
    }
    max_dev = {"phi": 0.0, "psi": 0.0}

    for key in sorted(data.keys()):
        traj = np.asarray(data[key])
        if traj.ndim != 2 or traj.shape[1] < 2:
            raise ValueError(
                f"Array '{key}' must have shape (n_frames, 2+) for phi/psi columns; got {traj.shape}."
            )

        phi = traj[:, 0]
        psi = traj[:, 1]

        phi_sin = np.sin(phi)
        phi_cos = np.cos(phi)
        psi_sin = np.sin(psi)
        psi_cos = np.cos(psi)

        phi_dev = np.max(np.abs(phi_sin * phi_sin + phi_cos * phi_cos - 1.0))
        psi_dev = np.max(np.abs(psi_sin * psi_sin + psi_cos * psi_cos - 1.0))
        max_dev["phi"] = max(max_dev["phi"], float(phi_dev))
        max_dev["psi"] = max(max_dev["psi"], float(psi_dev))

        n_chunks = traj.shape[0] // chunk_size
        if n_chunks == 0:
            raise ValueError(
                f"Array '{key}' has only {traj.shape[0]} frames, smaller than chunk_size={chunk_size}."
            )
        n_trim = n_chunks * chunk_size

        feature_chunks["phi_sin"].append(phi_sin[:n_trim].reshape(n_chunks, chunk_size))
        feature_chunks["phi_cos"].append(phi_cos[:n_trim].reshape(n_chunks, chunk_size))
        feature_chunks["psi_sin"].append(psi_sin[:n_trim].reshape(n_chunks, chunk_size))
        feature_chunks["psi_cos"].append(psi_cos[:n_trim].reshape(n_chunks, chunk_size))

    all_features = {
        name: np.concatenate(chunks, axis=0).astype(np.float32)
        for name, chunks in feature_chunks.items()
    }

    chosen = selected_features(dihedral_mode)
    return {name: all_features[name] for name in chosen}, max_dev


def to_multi_sim(chunks: np.ndarray, chunk_size: int) -> np.ndarray:
    """
    Convert (num_sims, chunk_size) chunks into scalar NF-TSF multi-sim format:
      shape (chunk_size, 1 + num_sims)
      col 0 = frame index, col 1+ = simulation trajectories.
    """
    time_col = np.arange(chunk_size, dtype=np.float32).reshape(-1, 1)
    return np.hstack([time_col, chunks.T.astype(np.float32)])


def main() -> None:
    args = parse_args()
    validate_args(args)

    rng = np.random.default_rng(args.seed)
    features = selected_features(args.dihedral)

    print(f"Loading input:          {args.input}")
    print(f"Selected dihedral mode: {args.dihedral}")
    print(f"Selected features:      {features}")
    print(f"Chunk size:             {args.chunk_size}")
    print(f"Train fraction:         {args.train_frac}")
    print(f"Random seed:            {args.seed}")

    chunked_features, unit_norm_deviation = splice_circular_features(
        args.input, args.dihedral, args.chunk_size
    )

    n_chunks_total = next(iter(chunked_features.values())).shape[0]
    for fname, arr in chunked_features.items():
        if arr.shape[0] != n_chunks_total:
            raise RuntimeError(
                f"Feature '{fname}' has mismatched chunk count {arr.shape[0]} != {n_chunks_total}."
            )

    print(f"Total chunks:           {n_chunks_total}")

    idx = rng.permutation(n_chunks_total)
    n_train = int(n_chunks_total * args.train_frac)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    print(f"Train chunks:           {len(train_idx)}")
    print(f"Test chunks:            {len(test_idx)}")

    if args.dihedral in ("phi", "both"):
        print(
            "Circular unit-norm check (phi): "
            f"max|sin^2+cos^2-1| = {unit_norm_deviation['phi']:.3e}"
        )
    if args.dihedral in ("psi", "both"):
        print(
            "Circular unit-norm check (psi): "
            f"max|sin^2+cos^2-1| = {unit_norm_deviation['psi']:.3e}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    train_shapes = []
    test_shapes = []

    for feature_name in features:
        chunks = chunked_features[feature_name]
        train_chunks = chunks[train_idx]
        test_chunks = chunks[test_idx]

        train_data = to_multi_sim(train_chunks, args.chunk_size).astype(np.float32)
        test_data = to_multi_sim(test_chunks, args.chunk_size).astype(np.float32)

        train_path = os.path.join(args.output_dir, f"{args.prefix}_{feature_name}_train.npy")
        test_path = os.path.join(args.output_dir, f"{args.prefix}_{feature_name}_test.npy")

        np.save(train_path, train_data)
        np.save(test_path, test_data)

        train_shapes.append(train_data.shape)
        test_shapes.append(test_data.shape)

        print(f"Saved train: {train_path}")
        print(f"  shape={train_data.shape}, dtype={train_data.dtype}")
        print(f"Saved test:  {test_path}")
        print(f"  shape={test_data.shape}, dtype={test_data.dtype}")

    if len(set(train_shapes)) != 1:
        raise RuntimeError(f"Train output shapes are inconsistent: {train_shapes}")
    if len(set(test_shapes)) != 1:
        raise RuntimeError(f"Test output shapes are inconsistent: {test_shapes}")

    print(f"All train output shapes match: {train_shapes[0]}")
    print(f"All test output shapes match:  {test_shapes[0]}")

    print("\nExamples:")
    print("  python splice_alanine_circular.py --dihedral phi --chunk_size 1000 --output_dir ./data")
    print("  python splice_alanine_circular.py --dihedral psi --chunk_size 1000 --output_dir ./data")
    print("  python splice_alanine_circular.py --dihedral both --chunk_size 1000 --output_dir ./data")


if __name__ == "__main__":
    main()
