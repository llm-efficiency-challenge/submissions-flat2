# How to test HELM locally

## Install the NeurIPS client

Install HELM: `pip install git+https://github.com/stanford-crfm/helm.git`


## Setup an HTTP server

Through this you can setup your http server:
`uvicorn main:app --host 0.0.0.0 --port 8080`

## Configure HELM

You can configure which datasets to run HELM on by editing a `run_specs.conf`, to run your model on a large set of datasets, take a look at https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/presentation/run_specs_lite.conf for some inspiration

Here's how you can create a simple one for the purposes of making sure that your model works

```bash
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
helm-summarize --suite v1
```

## Run HELM

If you need to do your test locally, you can change the 'model' in your conf. Remember to write in your yamls, ./prod_env/model_deployments.yaml and ./prod_env/tokenizer_configs.yaml

## Analyze your results

You can launch a web server to visually inspect the results of your run, `helm-summarize` can also print the results textually for you in your terminal but we've found the web server to be useful.

```
helm-server
```
