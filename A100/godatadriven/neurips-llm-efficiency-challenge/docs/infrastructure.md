{% include "../infra/README.md" %}

## MLFlow skinny

The 'mlflow.Dockerfile' builds a skinny version of MLflow that can be used when you e.g. want to download artifacts in init containers and don't need all the extras.

### Building & pushing

### Debugging

To debug, use:

```
docker run -it \
    --entrypoint /bin/bash \
    -e MLFLOW_TRACKING_URI=http://34.90.207.177:5000 \
    --mount type=bind,src="$(pwd)/.secrets/neurips-llm-eff-challenge-2023-<RANDOM-HASH>.json,target=/etc/credentials/sa.json" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/etc/credentials/sa.json \
    neurips/mlflow-skinny-gcp
```

Then, e.g.

```
mlflow artifacts download -r <RUN-ID> -a model -d .
```
