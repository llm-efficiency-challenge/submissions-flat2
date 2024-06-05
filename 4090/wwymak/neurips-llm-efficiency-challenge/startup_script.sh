#! /bin/bash

conda create -n neurips2023-1 python=3.10 --yes && conda activate neurips2023-1
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118

cd lit-gpt && pip install -r requirements.txt
pip install huggingface_hub
