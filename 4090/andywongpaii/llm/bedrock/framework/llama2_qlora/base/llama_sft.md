1. **Initialize docker**

Option 1: Work directly on base docker
```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/workspace/$(basename $(pwd)) --rm nvcr.io/nvidia/pytorch:22.12-py3
# to execute python code directly
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/workspace/$(basename $(pwd)) --rm nvcr.io/nvidia/pytorch:22.12-py3 python ./$(basename $(pwd))/train.py
```

Option 2: Build and run a stable docker
```bash
docker build -t llama_sft .
# docker run --gpus all -p 8080:80 llama_sft
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it llama_sft /bin/bash
# for development
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/workspace/$(basename $(pwd)) --rm llama_sft /bin/bash
```

verify pytorch installation
```python
import torch;
out = torch.rand(2, 3, device='cuda') @ torch.rand(3, 2, device='cuda')
print(torch.__version__)
print(out)
exit()
```

2. [solved]: Encountered issues when `pip bitsandbytes==0.40.1`.<br>
Here we decided to `pip bitsandbytes==0.41.1` via `dependencies.txt` during docker image building.
Alternative solution is to complie from source, following the insturction on 
[github](https://github.com/TimDettmers/bitsandbytes/issues/666).

<details>
  <summary>&nbsp;&nbsp;&nbsp;&nbsp;Details</summary>

Through github, I could not checkout bitsandbytes==0.40.1, could try 0.41 instead.
```bash
git clone https://github.com/timdettmers/bitsandbytes.git
cd bitsandbytes
git checkout 0.41.0
```
on image:
```bash
CUDA_VERSION=118 make cuda11x
cd bitsandbytes
python setup.py develop
export CUDA_HOME=/share/nvhpc/Linux_x86_64/22.11/cuda/11.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME
python -m bitsandbytes
if anything goes wrong, search bitsandbytes/bitsandbytes/cuda_setup/main.py 's code for bug.
see whether “binary_path = package_dir / self.binary_name” exists.
In my case, it is "bitsandbytes/bitsandbytes/libbitsandbytes_cuda118.so"
```

</details>
<br>

3. [Solved]: Encountered issue on `import peft`, due to failure on `import transformer_engine.tensorflow as te`.<br>
[pip - from Github](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html).
This approach takes a long time to build the installation wheel. Recommend sharing this final docker image to Docker Hub later.
```bash
# optional (have not been tested):
export NVTE_FRAMEWORK=torch
# without explicitly specify frameworks, building wheel took a while.
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```

4. [Solved]: Encountered issue on `import transformers.trainer`.<br>
Need `pip uninstall -y apex` as suggested by [github issue](https://github.com/huggingface/transformers/issues/22197)

5. Execute SFT example
```bash
cd PPO_learning
python NDEE/sft/databricks/test_fine_tune_qlora.py
```

Future:

<details>
  <summary>**To share docker image without building from scratch.**</summary>
Docker Hub (or other registries):

The most common way to share a Docker image is to push it to a container registry like Docker Hub, Google Container Registry (GCR), Amazon Elastic Container Registry (ECR), etc. Once it's on a registry, others can simply pull the image using docker pull.

Steps:

Log in to Docker Hub: docker login
Tag your image: docker tag your-image-name:your-tag username/repository:tag
Push your image: docker push username/repository:tag
Share the image name with others: username/repository:tag
Others can pull the image: docker pull username/repository:tag
Export and Import:

If you don't want to use a container registry, you can export the Docker image as a tar file, share that file, and then others can import it.

Steps:

Export the image: docker save -o image-name.tar image-name:tag
Share the image-name.tar file using any method (e.g., file transfer, cloud storage).
Others can import the image: docker load -i image-name.tar
</details>