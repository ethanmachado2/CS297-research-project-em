"""Install the single requirements file while preserving managed GPU/Jupyter packages.

Run with the notebook kernel's interpreter before importing project/ML modules.
Package versions remain defined only in requirements.txt. Exact versions of the
runtime's Torch/Torchvision are constrained so transitive dependencies cannot
silently replace a working CUDA build.
"""
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
import tempfile


MANAGED_PACKAGES = {"torch", "torchvision", "ipykernel", "jupyter", "nbclient", "nbformat"}


def cloud_requirements(text):
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        name = re.split(r"[<>=!~\[;\s]", stripped, maxsplit=1)[0].lower()
        if stripped and not stripped.startswith("#") and name not in MANAGED_PACKAGES:
            lines.append(stripped)
    return "\n".join(lines) + "\n"


def main():
    project = Path(__file__).resolve().parents[1]
    constraints = []
    for name in ("torch", "torchvision"):
        try:
            constraints.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            if name == "torch":
                raise RuntimeError("Choose a PyTorch GPU runtime before installing cloud dependencies")
    with tempfile.TemporaryDirectory() as temporary:
        requirements = Path(temporary) / "requirements.txt"
        requirements.write_text(cloud_requirements((project / "requirements.txt").read_text()))
        constraint_file = Path(temporary) / "constraints.txt"
        constraint_file.write_text("\n".join(constraints) + "\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements),
                        "-c", str(constraint_file)], check=True)
    print("Dependencies installed. If pip requests a kernel restart, restart before continuing.")


if __name__ == "__main__":
    main()
