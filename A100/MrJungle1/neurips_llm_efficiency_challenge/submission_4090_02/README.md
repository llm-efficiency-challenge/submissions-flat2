# Run Training And Inference Docker Locally

To build and run the taining Docker we need to execute:

```bash
docker build -f ./Dockerfile.train -t submission_4090_02_train .

docker run --gpus "device=0" --rm -ti submission_4090_02_train
```

The inference Docker is created and started with:

```bash
docker build -f ./Dockerfile.inference -t submission_4090_02_inference .

docker run --gpus "device=0" -p 8080:80 --rm -ti submission_4090_02_inference
```

To test the inference docker we can run this query:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital of france? "}' http://localhost:8080/tokenize
OR
curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital of france? "}' http://localhost:8080/process
```
