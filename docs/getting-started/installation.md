# Installation

The default install is the complete Qorona: every runtime dependency needed to use every
feature. Python 3.11 or newer is required.

## With pip

Installs the latest release and gives you the `qorona` command:

```bash
pip install qorona
```

## With conda (development checkout)

Clone the repository and build the provided environment from its root. This gives you the
development version (`main`) as an editable checkout, e.g. to develop or to add a model reader:

```bash
git clone https://github.com/RayanDhib/Qorona.git
cd Qorona
conda env create -f environment.yml
conda activate qorona
```

## On a cluster

Qorona installs from PyPI into a plain virtual environment; no conda needed. The
[HPC clusters guide](hpc.md) covers modules, SLURM templates, and the flags made for
batch jobs.
