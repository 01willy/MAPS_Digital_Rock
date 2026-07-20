#!/usr/bin/env python3
"""Convert a downloaded Micro-CT volume to the MAPS input format.

Target format expected by ``maps.data.load_volume`` and the training/eval
drivers:

    raw ``float32`` binary, shape 1000 x 1000 x 1000, axis order Z, Y, X
    (C-contiguous), values in [0, 1] with 0 = pore and 1 = solid.

The eight public source volumes (Digital Rocks Portal project 317; Imperial
College London pore-scale collection, see the README Data section) are
distributed as segmented or grayscale 8/16-bit raw or ``.mhd``/``.raw`` pairs.
This helper reads the common variants, normalizes to [0, 1], sets the
pore/solid polarity, and writes the ``.bin`` MAPS expects.

Examples
--------
# Segmented uint8 already 0=pore/1=solid (or 0/255), native Z,Y,X:
python scripts/prepare_data.py in.raw data/Bentheimer_1000c_f32.bin \
    --shape 1000 1000 1000 --dtype uint8

# Grayscale uint16 with a threshold (values >= T are solid):
python scripts/prepare_data.py in.raw data/Ketton_1000c_f32.bin \
    --shape 1000 1000 1000 --dtype uint16 --threshold 32768

# Source labels 0=solid/1=pore (invert polarity):
python scripts/prepare_data.py in.raw out.bin --dtype uint8 --pore-label 1 \
    --solid-is-high

Notes
-----
* If the source axis order is X,Y,Z (some portal exports), pass ``--transpose
  2 1 0`` to reorder to Z,Y,X.
* This script does not segment grayscale scans beyond a single global
  threshold; use the source's provided segmentation where available, which is
  what the paper uses.
"""
import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a Micro-CT volume for MAPS.")
    ap.add_argument("src", help="input raw/.raw file (headerless) or .npy")
    ap.add_argument("dst", help="output float32 .bin (Z,Y,X, [0,1], 0=pore/1=solid)")
    ap.add_argument("--shape", type=int, nargs=3, default=[1000, 1000, 1000],
                    metavar=("Z", "Y", "X"), help="volume shape (default 1000 1000 1000)")
    ap.add_argument("--dtype", default="uint8",
                    choices=["uint8", "uint16", "float32"], help="input dtype")
    ap.add_argument("--transpose", type=int, nargs=3, default=None,
                    metavar=("A", "B", "C"),
                    help="axis permutation to reach Z,Y,X (e.g. 2 1 0 for X,Y,Z input)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="grayscale threshold; values >= T become solid")
    ap.add_argument("--pore-label", type=int, default=0,
                    help="label value that denotes pore in a segmented input (default 0)")
    ap.add_argument("--solid-is-high", action="store_true",
                    help="for grayscale/threshold, high intensity is solid (default)")
    args = ap.parse_args()

    if args.src.endswith(".npy"):
        vol = np.load(args.src)
    else:
        vol = np.fromfile(args.src, dtype=np.dtype(args.dtype))
        n = int(np.prod(args.shape))
        if vol.size != n:
            raise SystemExit(f"size mismatch: file has {vol.size} voxels, "
                             f"shape {tuple(args.shape)} expects {n}")
        vol = vol.reshape(args.shape)

    if args.transpose is not None:
        vol = np.transpose(vol, args.transpose)

    vol = vol.astype(np.float32)

    if args.threshold is not None:
        solid = vol >= args.threshold
    else:
        # segmented input: solid = everything that is not the pore label
        solid = vol != float(args.pore_label)
        if args.solid_is_high and vol.max() > 1.5:
            # explicit 0/255 or 0/high grayscale segmentation
            solid = vol >= (vol.max() / 2.0)

    out = solid.astype(np.float32)  # 0 = pore, 1 = solid
    out = np.ascontiguousarray(out)  # C order, Z,Y,X
    out.tofile(args.dst)
    phi = 1.0 - float(out.mean())
    print(f"wrote {args.dst}  shape={out.shape}  porosity phi={phi:.4f}  "
          f"(0=pore, 1=solid, float32)")


if __name__ == "__main__":
    main()
