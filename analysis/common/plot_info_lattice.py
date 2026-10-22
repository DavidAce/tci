#!/usr/bin/env python3
"""Plot a Sobol information lattice as staggered circular nodes.

HDF5 input: dataset tci/sil (row=scale, column=start position).
Alternatively, read a whitespace-separated text matrix or CSV.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, PowerNorm
import numpy as np


def read_info_lattice(path, dsetpath):
    if Path(path).suffix.lower() in {'.h5', '.hdf5', '.hdf'}:
        import h5py
        with h5py.File(path, 'r') as f:
            il = np.asarray(f[dsetpath], dtype=np.float64)
    else:
        il = np.loadtxt(path, delimiter=',' if path.suffix.lower() == '.csv' else None)

    if il.ndim != 2 or il.shape[0] != il.shape[1]:
        raise ValueError('Expected a square SIL matrix: rows=scale, columns=position')
    if not np.isfinite(il).all():
        raise ValueError('SIL contains nonfinite values')
    if np.min(il) < -1e-10 or np.max(il) > 1 + 1e-10:
        raise ValueError('SIL entries must lie in [0,1] (up to roundoff)')
    return np.clip(il, 0, 1)


def plot_info_lattice(il, *, logarithmic=False, vmin=1e-5, vmax=1.0, cmap='Greens', size=280):
    d = il.shape[0]
    # A node at interval [a, a+ell] is centered at a+ell/2.
    scale, start = np.indices((d, d))
    valid = (scale + start) < d
    ell, a = scale[valid], start[valid]
    x = a + ell / 2                   # zero-based variable indices
    y = ell * np.sqrt(3) / 2         # equilateral triangular lattice
    values = il[valid]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    colors = plt.get_cmap(cmap).with_extremes(bad='white', under='white')
    #vmin_abs = np.min(np.abs(values))
    vmin_nnz = np.min(values[np.nonzero(values)])
    if logarithmic:
        vmin = vmin_nnz * 0.1
    else:
        vmin=0

    norm = (LogNorm(vmin=vmin, vmax=vmax) if logarithmic
            else PowerNorm(0.22, vmin=0, vmax=vmax))
            # else Normalize(vmin=0, vmax=vmax))
    dots = ax.scatter(x, y, c=values, cmap=colors, norm=norm,
                      marker='o', s=size, edgecolors='black', linewidths=0.6)

    ax.set_aspect('equal')
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(-0.4, (d - 1) * np.sqrt(3) / 2 + 0.6)
    ax.set_xticks(np.arange(d))
    ax.set_yticks(np.arange(d) * np.sqrt(3) / 2,
                  labels=np.arange(d))
    ax.set_xlabel('Parameter index')
    ax.set_ylabel(r'Scale $\ell$')
    ax.spines[['top', 'right']].set_visible(False)
    fig.colorbar(dots, ax=ax, label='Value', shrink=0.75, pad=0.04)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='HDF5 file with an information lattice matrix')
    parser.add_argument('--dsetpath', default='tci/sil', help='HDF5 dataset path')
    parser.add_argument('--log', action='store_true', help='Logarithmic color scale')
    parser.add_argument('--vmin', type=float, default=1e-5, help='Logarithmic lower color limit')
    parser.add_argument('--vmax', type=float, default=1.0, help='Upper color limit')
    parser.add_argument('--cmap', default='Greens', help='Matplotlib colormap')
    parser.add_argument('--size', type=float, default=280, help='Node area in points squared')
    parser.add_argument('--output', type=Path, help='Save instead of showing (PNG/PDF/SVG)')
    args = parser.parse_args()

    il = read_info_lattice(args.file, args.dsetpath)
    if args.vmax <= 0 or (args.log and (args.vmin <= 0 or args.vmin >= args.vmax)):
        parser.error('Require vmax > 0 and, for --log, 0 < vmin < vmax')
    fig, _ = plot_info_lattice(il, logarithmic=args.log, vmin=args.vmin,
                               vmax=args.vmax, cmap=args.cmap, size=args.size)
    if args.output:
        fig.savefig(args.output, dpi=220)
    else:
        plt.show()


if __name__ == '__main__':
    main()
