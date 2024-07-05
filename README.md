<div align="center">

# symbolic_probing
<!-- 
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

What it does...

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/mh-amani/symbolic_probing
cd symbolic_probing

# [OPTIONAL] create conda environment
conda create -n symbolic_probing python=3.11
conda activate symbolic_probing

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/mh-amani/symbolic_probing
cd symbolic_probing

# create conda environment and install dependencies
conda env create -f environment.yaml -n symbolic_probing

# activate conda environment
conda activate symbolic_probing
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
