
set -e pipefail
sub=3
echo "running submission $sub....."
echo "starting to build docker image"
docker build -f ./Dockerfile$sub -t llama_recipes_inference .
echo "starting to run docker image"
docker run -it --gpus "device=0" -p 8080:80 llama_recipes_inference:latest
