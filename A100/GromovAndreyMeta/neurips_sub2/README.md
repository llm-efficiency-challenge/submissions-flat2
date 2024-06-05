# Docker Submission
These are the instructions to run the docker container.

### Make your GPUs visible to Docker 
Follow this guide to install [nvidia-ctk](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```sh
nvidia-ctk runtime configure
systemctl restart docker
```

### Build and run 
```sh
docker build -t sample_submission .
docker run --gpus all -p 8080:80 sample_submission
```
### Send requests
```sh
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "The capital of france is "}' http://localhost:8080/process
```
