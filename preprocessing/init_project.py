#!/usr/bin/env python3
"""Project environment management utilities."""

import subprocess
import sys
import shutil
from pathlib import Path

# --- Config (mirror your Makefile variables) ---
CONDA_ENV_NAME = "circ_milan"
PYTHON_VERSION = "3.11"
PYTHON_INTERPRETER = "python3"
VENV_DIR = "venv_circ_311"
MLFLOW_PORT = 5501


def create_conda_env():
    """Print instruction to create conda environment."""
    print("Run the following to create your conda environment:")
    print(f"\n  conda create -n {CONDA_ENV_NAME} python={PYTHON_VERSION}\n")


def create_venv():
    """Create a virtual environment."""
    subprocess.run([PYTHON_INTERPRETER, "-m", "venv", VENV_DIR], check=True)
    print(
        f"Virtual environment created at '{VENV_DIR}' using {PYTHON_INTERPRETER} {PYTHON_VERSION}"
    )


def activate_venv():
    """Print instructions to activate the virtual environment."""
    print("Run the following to deactivate conda and activate the venv:\n")
    print("  conda deactivate")
    print(f"  source {VENV_DIR}/bin/activate  # Unix/Mac")
    print(f"  {VENV_DIR}\\Scripts\\activate     # Windows\n")


def clean_venv():
    """Remove the virtual environment directory."""
    venv_path = Path(VENV_DIR)
    if venv_path.exists():
        shutil.rmtree(venv_path)
        print(f"Virtual environment '{VENV_DIR}' removed.")
    else:
        print(f"No virtual environment found at '{VENV_DIR}'.")


def install_requirements():
    """Upgrade pip and install requirements."""
    subprocess.run(
        [PYTHON_INTERPRETER, "-m", "pip", "install", "-U", "pip"], check=True
    )
    subprocess.run(
        [PYTHON_INTERPRETER, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )


def clean_pycache():
    """Delete all compiled Python files and __pycache__ dirs."""
    removed = 0
    for pattern in ["**/*.pyc", "**/*.pyo"]:
        for f in Path(".").rglob(pattern[3:]):  # strip **/ for rglob
            f.unlink()
            removed += 1
    for d in Path(".").rglob("__pycache__"):
        shutil.rmtree(d)
        removed += 1
    print(f"Cleaned {removed} compiled files/cache directories.")


def mlflow_ui():
    """Launch the MLflow UI."""
    subprocess.run(
        [
            "mlflow",
            "ui",
            "--backend-store-uri",
            "mlruns",
            "--host",
            "0.0.0.0",
            "--port",
            str(MLFLOW_PORT),
        ],
        check=True,
    )


COMMANDS = {
    "create_conda_env": create_conda_env,
    "create_venv": create_venv,
    "activate_venv": activate_venv,
    "clean_venv": clean_venv,
    "requirements": install_requirements,
    "clean": clean_pycache,
    "mlflow_ui": mlflow_ui,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("\nProject Environment Utilities")
        print("=" * 40)
        print("Usage:  python scripts/env_utils.py <command>\n")
        print("Available commands (run in this order):")
        print(
            "  1. create_conda_env  — print instructions to create a conda environment"
        )
        print("  2. create_venv       — create a virtual environment")
        print("  3. activate_venv     — print instructions to activate the venv")
        print("  4. requirements      — upgrade pip and install requirements.txt")
        print()
        print("Utility commands (run as needed):")
        print("  clean_venv        — remove the virtual environment")
        print("  clean             — remove all .pyc files and __pycache__ dirs")
        print("  mlflow_ui         — launch the MLflow UI on port 5501")
        print()
        print("Example:")
        print("  python scripts/env_utils.py requirements\n")
        print("NOTE: Equivalent 'make' commands are available for all of the above.")
        print("      Consult the README for full setup instructions, including")
        print("      how to use the Makefile targets and when to prefer one")
        print("      approach over the other.\n")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
