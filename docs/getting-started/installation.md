# Installation

Qorona needs Python 3.11 or newer. The default install has all the packages for the standard
products.

## With pip

Install the latest release. This gives you the `qorona` command:

```bash
pip install qorona
```

## With conda (development version)

Clone the repository and create the environment from its root. This gives the development version
(`main`) as an editable install:

```bash
git clone https://github.com/RayanDhib/Qorona.git
cd Qorona
conda env create -f environment.yml
conda activate qorona
```

## On a cluster

Install from PyPI into a plain virtual environment. You do not need conda. See
[HPC clusters](hpc.md).

Next: the [Quick start](first-eclipse-image.md).
