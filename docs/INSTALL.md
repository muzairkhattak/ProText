# Installation

### Acknowledgement: This readme file for installing datasets is modified from [PromptSRC's](https://github.com/muzairkhattak/PromptSRC) official repository.

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n protext python=3.8

# Activate the environment
conda activate protext

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
* Clone ProText code repository and install requirements
```bash
# Clone PromptSRC code base
git clone https://github.com/muzairkhattak/ProText.git

cd ProText/
# Install requirements

pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```

* Install dassl library.

**Note:** We have modified original dassl library in order to configure text-based data-loaders for training. Therefore it is strongly recommended to directly utilize [Dassl library provided in this repository](../Dassl.pytorch).
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# No need to clone dassl library as it is already present in this repository
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

