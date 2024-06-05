# Neurips 1 LLM 1 GPU Challenge

This repository contains all work related to our submission for the [2023 Neurips LLM efficiency challenge](https://llm-efficiency-challenge.github.io/).

## Team members

- Jasper Ginn (lead): jasper.ginn@xebia.com
- Jetze Schuurmans
- Rens Dimmerdaal
- Caio Benatti
- Yke Rusticus

## Dataset

The dataset that we used to train the model can be found [here](https://huggingface.co/datasets/JasperHG90/neurips-efficiency-challenge-2023).

## Training

Our training configuration can be found in:

1. jobs/training/mistral7b_4090.job_config.yml for the 4090 track
1. jobs/training/mistral7b_A100.job_config.yml for the A100 track

We use a CLI for training, execute `cajajejo models finetune-model --help` for more information.

## Submission (model)

The files for our submission files can be found in the 'submission' folder. The submission dockerfile is called 'Dockerfile'.

We have also added a zipfile containing all relevant files. This archive is called 's3-1c34595f2ffa4cd5bdc9e98c1d206dda.zip'.

### Building the submission ZIP

Run:

1. `poetry export --without-hashes -E training -E submission -o submission/requirements.txt`
1. `rm -R dist/* && poetry build`
1. `rm -R submission/dist && cp -R dist submission`
1. `zip Dockerfile submission submission.zip`

### Debugging the submission server

To debug the submission server, you can use the following command:

```
docker run -p 8080:80 -it --entrypoint cajajejo  neurips/submission api-server start /app/config.yml --host 0.0.0.0 --port 80 --dry-run
```

This will boot the submission server with a small pythia model.

## Installing the python project

NB: installing this project requires poetry>=1.5.1

For a fresh installation of the project, use:

```
poetry lock
poetry install --with dev
```

There are two additional extras:

- training: everything to do with training models. Only possible to install these dependencies on Linux because of pinned torch nightly version. You can also use the 'training.Dockerfile'.
- evaluation: everything to do with evaluating models. You can also use the 'evaluation.Dockerfile'.

### Versioning

We use the 'poetry-git-version-plugin' for versioning the `cajajejo` source code. This plugin allows us to version code by git commit tags when building the project using `poetry build` (and hence gives us a means to track which version was used for jobs). You should install it using:

```
poetry self add poetry-git-version-plugin
```

### PyInvoke commands

There are two 'PyInvoke' scripts that you can use to manage infrastructure or do perform common tasks. The first of these is stored in 'tasks.py' and contains the following commands (execute `poetry run invoke --list` to see the full list):

```text
  authenticate-kubectl          🔑 Authenticate GKE cluster in kubectl
  build-and-push-helm-image     🐳+🌬 Build and push mlflow image to GCP docker registry
  build-and-push-job-image      🐳+🌬 Build and push evaluation image to GCP docker registry
  build-and-push-mlflow-image   🐳+🌬 Build and push mlflow image to GCP docker registry
  build-helm-image              🐳 Build helm image
  build-job-image               🐳 Build docker image needed for evaluation or training job
  build-mlflow-image            🐳 Build the mlflow image
  docker-login                  🐳 Log in to the GCP docker registry
  get-job-logs                  📜 Get logs for a training or evaluation job
  install-dependencies          ⬇ Install project dependencies and optional dependencies
  push-helm-image               🌬 Push helm image to GCP docker registry
  push-job-image                🌬 Push job image to GCP docker registry
  push-mlflow-image             🌬 Push mlflow image to GCP docker registry
  serve-docs                    📓 Serve the documentation locally
  submit-evaluation-job         🚀 Submit evaluation job to GKE cluster
  update-pip                    ☝ This will update pip in the virtual environment
```

The second PyInvoke script can be found in the 'infra' directory, and contains the following commands:

```text
  get-gcp-project-number             ⅐ Get the GCP project number or retrieve it from env variables
  tf-apply                           🚀 Run terraform apply
  tf-docs                            📕 Generate terraform docs
  tf-plan                            📜 Run terraform plan
  write-gcp-compute-ssh-key          📝 Write ssh key for gcp compute
  write-json-key-dvc-storage-admin   📝 Print json key for dvc storage admin GCP service account
```

For help on a specific command, execute `poetry run invoke <command> --help`.

## Docs

To generate the documentation, install the poetry project and execute:

1. `poetry install --with docs ` (The docs dependency is not installed by default, and you need extras for evaluation & training)
1. `poetry shell`
1. `invoke serve-docs`
