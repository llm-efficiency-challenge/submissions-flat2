 llm

## 1. Getting Started
This repo has only works on Linux environment.<br>
If you use windows OS, I am trying to figure out on how to use [WSL-2](https://learn.microsoft.com/en-us/windows/wsl/install) with deepspeed.

### Hardware Prerequisite:
1. 32+ GB DRAM Memory
2. 24+ GB GPU Memory
3. Linux or Linux WSL-Ubuntu 2.0 (Windows)

### Software Prerequisite:
As of September 2, 2023, DeepSpeed-Chat seems to have compatibility issue with torch 2.0.1.<br>

### Linux WSL-Ubuntu 2.0 set up
To properly setup cuda 11.8 on WSL, follow the instruction on [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [ubuntu's guide to enable GPU acceleration](https://ubuntu.com/tutorials/enabling-gpu-acceleration-on-ubuntu-on-wsl2-with-the-nvidia-cuda-platform#1-overview).<br>
Also follow this page to download [cuda 11.8 for WSL-Ubuntu](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local).<br>
```sh
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
rm cuda_11.8.0_520.61.05_linux.run
```
To test the installation, try:
```sh
nvcc --version
```

❗ If return errors `Command 'nvcc' not found, but can be installed with: sudo apt install nvidia-cuda-toolkit`, try this **FIRST**: 
```sh
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
nvcc --version
```

If sucessfull, permenately update the ~/.bashrc file
```sh
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc
echo 'export PATH=${CUDA_HOME}/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

If encountered trouble installing cuda with `deb (local)`, try installing cuda through `runfile (local)`, as recommended by [NVCC issue](https://forums.developer.nvidia.com/t/wsl-version-of-toolkit-doesnt-install-nvcc-cant-get-cuda-to-work-on-wsl/227214).

### 1a. clone this repo
For **development purpose**, consider recursively clone the top this repository in order to ensure everything within the repo is fully functional. 
```sh
git clone --recurse-submodules git@github.com:andywongpaii/llm.git
```

### 1b. set up conda environment (cuda 11.7, pytorch1.13)
HELM and Deepspeed seems to have incompatible package version dependency.<br>
Since inference and evaluation server are separated anyway, mind as well creating two conda environment, with `neurips2023-eval` and `neurips2023-train`.
NOTE❗: <br>
I encountered inference issue with deepspeed using cuda 11.8 + torch2.0 + RTX4090x1,<br>
so I decided to try working with cuda 11.7 + torch1.13.1 + RTX1080Tix4, with `neurips2023-train_SAFE` for now.
Before installing deepspeed, make sure libaio is installed by `sudo apt install libaio-dev`.<br>
Before installing deepspeed, must install pytorch first.<br>
Check if there are any issue on installing deepspeed by executing `ds_report`, user may ignore the `sparse_attention` status.

```sh
conda create -n neurips2023-train python=3.9 --yes
conda activate neurips2023-train
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements-train.txt
pip install -e .
ds_report

conda create -n neurips2023-train_SAFE python=3.9 --yes
conda activate neurips2023-train_SAFE
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements-train_SAFE.txt
pip install -e .
ds_report

conda create -n neurips2023-train_QLoRA python=3.9 --yes
conda activate neurips2023-train_QLoRA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements-train_QLoRA.txt
pip install -e .

conda create -n neurips2023-eval python=3.9 --yes
conda activate neurips2023-eval
pip install -r requirements-eval.txt
# if encountered issue, try crfm-helm.
pip install -e .
helm-run -h
```
## 2. Evaluation Workflow
### 2a. Setup an HTTP server

#### NIPS toy example

1. initialize a docker:
Open ```Docker Desktop``` software on windows.<br>
Then, on wsl terminal...<br>
<br>
For the first time, build and initiate docker image:
```sh
cd bedrock/integrated/neurips_llm_efficiency_challenge/toy-submission
docker build -t toy_submission .
docker run --gpus all -p 8080:80 toy_submission
```
If sucessful, there should be a messeage stating `INFO: Uvicorn running on http://0.0.0.0:80`.<br>

If docker setup were sucessful, simply `stop` or `start` the `toy_submission` container on the ```Docker Desktop``` interface.

2. On a separate terminal, test the model by submitting requests:

```sh
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "The capital of france is "}' http://localhost:8080/process
```
There should be a response starting with something like `{"text":"The capital of france is the city of paris. the french capital has been described by many as one of the most beautiful cities in the world. it is a very popular destination for tourists around the world.\nAccording to the latest travel surveys, a lot of","tokens"...,"request_time":266.0}`


### 2b. Evaluate model with HELM

You can configure which datasets to run HELM on by editing a `run_specs.conf`.<br>
An example of lite evaluation can be found at ```bedrock/evaluation/helm/src/helm/benchmark/presentation/run_specs_lite.conf```

## 3. Experiments

### 3a. infer and benchmark Llama2
```sh
# activate environment
conda activate neurips2023-train
conda activate neurips2023-train_SAFE
# check deespeed setup status
ds_report
# hugging face inference: (check if `gpt-neo-125m` and 'EleutherAI/gpt-neo-2.7B' works.)
python NIPS_llm_efficiency_challenge_submission/2023/test/test-hf_models.py
# deepspeed inference: Google-T5_small
python NIPS_llm_efficiency_challenge_submission/2023/test/test-T5_small.py
# deepspeed inference: gpt_neo 125m
python NIPS_llm_efficiency_challenge_submission/2023/test/test-gpt_neo.py
# deepspeed inference: gpt_neo 2.7b
# OOM error on 1080ti
python NIPS_llm_efficiency_challenge_submission/2023/test/test-gpt_neo.py
```
**TODO**: 
1. [quantized-models-tutorials](https://www.deepspeed.ai/tutorials/inference-tutorial/#datatypes-and-quantized-models)
2. multi-gpu inferences
3. [BERT tutorial](https://www.philschmid.de/bert-deepspeed-inference)

#### 7B
#### 7B Chat
#### 7B Chat quantized
#### 7B Chat Wanda-pruned
#### 13B Chat quantized/Wanda-pruned
#### 70B Chat quantized/Wanda-pruned

### 3b. Deepspeed-chat

### 3c. 

#### To execute an evaluation:

```bash
conda activate neurips2023-eval
helm-run --conf-paths bedrock/evaluation/NIPS_test_bench/sample_run_spec.conf --suite v1.1 --max-eval-instances 10 #1000
helm-summarize --suite v1.1
```
Since the benchmark was execute at the home directory (`llm/`), the results will be stored at the home directory (e.g. `llm/benchmark_output` and `llm/prod_env` )

#### Analyze result on web interface
```bash
helm-server
```

This will launch a server on your local host (e.g. [http://localhost:8000](http://localhost:8000) ), if you're working on a remote machine you might need to setup port forwarding. If everything worked correctly you should see a page that looks like [this](https://user-images.githubusercontent.com/3282513/249620854-080f4d77-c5fd-4ea4-afa4-cf6a9dceb8c9.png)

### TODO:
1. ...

### Helpful resources:
1. [lightning_ai_to_get_started](https://lightning.ai/pages/community/tutorial/neurips2023-llm-efficiency-guide/)

### Notes:
1. deepspeed cache: `/home/ndeewong/.cache/torch_extensions/py39_cu118`
2. hugging face tmp data files: `/tmp/data_files/`

### Plan:
2. infer Llama-2 chat and pre-train for score.
modify from [1littlecoder](https://colab.research.google.com/drive/14GQw8HW8TllB_S3enqotM3dXU7Pav9e_?usp=sharing)
3. quantize large llama-2 chat and pre-train for inference and score. 
modify from [ludwig](https://colab.research.google.com/drive/1Ly01S--kUwkKQalE-75skalp-ftwl0fE?usp=sharing), [databricks](https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms), [Medium](https://abvijaykumar.medium.com/fine-tuning-llm-parameter-efficient-fine-tuning-peft-lora-qlora-part-1-571a472612c4), [Medium part2](https://abvijaykumar.medium.com/fine-tuning-llm-parameter-efficient-fine-tuning-peft-lora-qlora-part-2-d8e23877ac6f), [Medium other](https://ukey.co/blog/finetune-llama-2-peft-qlora-huggingface/), [TRL](https://huggingface.co/blog/trl-peft).
4. Enable [Flash attention](https://discuss.pytorch.org/t/flash-attention/174955/14) or [Author's implementation](https://github.com/Dao-AILab/flash-attention)
See [instruction-tune-llama-2](https://www.philschmid.de/instruction-tune-llama-2) for example.
```python
    torch.cuda.synchronize()
    start = time.time()
    for i in range(num_trials):
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=dropout_rate, training=True)
        x = (attn @ v).transpose(1, 2)  # .reshape(bz, seq_len, n_heads*dims)
    torch.cuda.synchronize()
    end = time.time()
    print('Standard attention took {} seconds for {} trials'.format(end - start, num_trials))

    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_trials):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_rate)
        torch.cuda.synchronize()
        end = time.time()
        print('Flash attention took {} seconds for {} trials'.format(end - start, num_trials))
```
4. estiablish fine-tune workflow with deepspeed using databriks-dolly-15
5. create my own mixture of data

## Acknowledgement and Citation 

We thank the following papers and open-source repositories:

    [1] ..., ..., et al. "...", https://... (20##).