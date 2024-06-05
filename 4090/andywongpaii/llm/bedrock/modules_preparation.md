# Documentation on setting up open sources module as LLM system sub-modules

### Submodules in bedrock
#### General Procedure
1. fork a repo on github.com
2. cd to `bedrock`/`your_submodule_category`
3. git submodule `your_forked_repo`

```sh
cd bedrock/`your submodule category`
git submodule add `your_fork_url`
```


#### integrated: [2023 NIPS efficeincy challenge](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge)
Fork: [neurips_llm_efficiency_challenge](https://github.com/andywongpaii/neurips_llm_efficiency_challenge/tree/master)<br>
```sh
cd bedrock/integrated
git submodule add https://github.com/andywongpaii/neurips_llm_efficiency_challenge.git
```


#### integrated: [llm_efficiency_challenge](https://github.com/siddheshmhatre/llm_efficiency_challenge)
Fork: None <br>
NOTE: direct modficiation as main contributor.
```sh
cd bedrock/integrated
git submodule add https://github.com/siddheshmhatre/llm_efficiency_challenge.git
cd bedrock/integrated/neurips_llm_efficiency_challenge/toy-submission
git submodule update --init --recursive
```


#### evaluation: [HELM (official)](https://github.com/stanford-crfm/helm).
Fork: [HELM](https://github.com/andywongpaii/helm)
NOTE: NIPS have an outdated fork at [NIPS_helm](https://github.com/drisspg/helm/tree/neruips_client). As of 8/23/2023, the fork has been merged with the official HELM, so techinically it should work using the official fork.
```sh
git submodule add https://github.com/andywongpaii/helm.git bedrock/evaluation/helm
```


#### framework: [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples/tree/master).
Fork: [DeepSpeedExamples](https://github.com/andywongpaii/DeepSpeedExamples.git)
```sh
git submodule add https://github.com/andywongpaii/DeepSpeedExamples.git bedrock/framework/DeepSpeedExamples
```

#### framework: [deepspeed-examples](https://github.com/microsoft/DeepSpeed).
Fork: [DeepSpeed](https://github.com/andywongpaii/DeepSpeed.git)
```sh
git submodule add https://github.com/andywongpaii/DeepSpeed.git bedrock/framework/DeepSpeed
```


#### framework: [lit-gpt]().
Fork: Todo
```sh
cd bedrock/framework
git submodule add `TODO`
```


#### TODO:
1. Sophia
2. Wanda Pruning
3. [hugging face modules](https://huggingface.co/docs/transformers/perf_train_gpu_one)
4. Llama 2

<br>
<br>
<br>

### Set up work environment
#### Nvidia
On Windows 11 + WSL:
TODO

#### Docker
On Windows 11 + WSL:
TODO

