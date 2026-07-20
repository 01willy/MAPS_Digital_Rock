#!/usr/bin/env python3
"""Poiseuille (parallel-plate) validation of the D3Q19 LBM solver.

Runs the analytical-channel check described in the paper (Section on metrics):
a parallel-plate channel of width W must reproduce k = W^2 / 12 in lattice
units. The paper reports agreement within 2.3% at W = 22 lattice widths.

Usage:
    python lbm/poiseuille_validation.py [--width 24] [--cpu]
Equivalent to `python lbm/d3q19.py --validate`.
"""
import argparse
import json

try:
    from lbm.d3q19 import validate_poiseuille
except ImportError:  # invoked as a script from within lbm/
    from d3q19 import validate_poiseuille


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--width', type=int, default=24,
                    help='channel width W in lattice units')
    ap.add_argument('--n_steps', type=int, default=5000)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    device = 'cpu' if args.cpu else 'cuda'
    result = validate_poiseuille(width=args.width, n_steps=args.n_steps,
                                 device=device)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
