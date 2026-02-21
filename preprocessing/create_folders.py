#!/usr/bin/env python3
"""Create standard project folder structure."""

from pathlib import Path

# Mirror the OUTCOMES variable from the Makefile
OUTCOMES = ["Bleeding_Edema_Outcome"]


def create_folders():
    dirs = [
        "data/external",
        "data/interim",
        "data/processed",
        "data/raw",
        "data/processed/inference",
        "models/results",
        "models/eval",
        "modeling",
        "preprocessing",
        "core",
    ]

    # Add per-outcome subdirectories
    for outcome in OUTCOMES:
        dirs.append(f"models/results/{outcome}")
        dirs.append(f"models/eval/{outcome}")

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")

    gitkeeps = [
        "data/interim/.gitkeep",
        "data/processed/.gitkeep",
        "data/processed/inference/.gitkeep",
        "models/results/.gitkeep",
        "models/eval/.gitkeep",
    ]

    inits = [
        "modeling/__init__.py",
        "preprocessing/__init__.py",
        "core/__init__.py",
    ]

    for f in gitkeeps + inits:
        Path(f).touch(exist_ok=True)
        print(f"Created: {f}")


if __name__ == "__main__":
    create_folders()
    print("\nProject structure ready.")
