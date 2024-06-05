
docker build -f ./Dockerfile -t llama_recipes_inference .

docker run --gpus "all" -p 8080:80 --rm -ti -v /data1/:/data1 llama_recipes_inference