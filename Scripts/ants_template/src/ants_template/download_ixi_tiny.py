"""Download IXITiny using TorchIO and print image paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download IXITiny dataset using TorchIO"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("ixi_tiny"),
        help="Root directory for the dataset",
    )
    args = parser.parse_args()

    import torchio as tio

    dataset = tio.datasets.IXITiny(root=args.root, download=True)
    paths = sorted(subject.image.path for subject in dataset)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
