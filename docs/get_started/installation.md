# Installing Gorget

## Prerequisites

Supported Python versions:

- 3.9
- 3.10
- 3.11

## Using `pip`

!!! note

    Consider installing the Gorget python packages on a virtual environment like `venv` or `conda`.

```bash
pip install gorget
```

If you have issue installing the package due to missing `torch`, you can try the following commands:

```bash
pip install wheel
pip install torch==2.0.1
pip install gorget --no-build-isolation
```

## Install from source

To install Gorget from source, first clone the repo:

- Using HTTPS
```bash
git clone https://github.com/Perruer/gorget.git
```
- Using SSH
```bash
git clone git@github.com:protectai/gorget.git
```

We recommend to use a virtual environment like `venv` or `conda` to install the package.

```bash
python -m venv venv
source venv/bin/activate
```

Then, install the package using `pip`:

```bash
python -m pip install ".[dev]"
```
