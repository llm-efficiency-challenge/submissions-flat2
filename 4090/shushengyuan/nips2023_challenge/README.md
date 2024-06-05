# nips2023_challenge

## Brief Description
We made 3 submissions placing in the `submissions/` folder, and the training codes are the same.

## Steps - evaluation
1. `cd submission/submission{N} && docker build -f ./Dockerfile -t inference .` (N=1, 2, 3)
2. `docker run --gpus "device=0" -p 8080:80 --rm -ti inference`


## Steps - finetune
1. `cd docker && DOCKER_BUILDKIT=0 docker build -t nips_challenge . && cd ..`
2. `bash scripts/finetune{N}.sh` (N=1, 2 ,3) 

## How to get our model
We use [dolly](https://huggingface.co/datasets/pankajmathur/dolly-v2_orca), [LIMA](https://huggingface.co/datasets/GAIR/lima), and [guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) respectively and use qlora combined with EMA(Exponential Moving Average) algorithm to obtain the final model, for more parameters see [here](scripts).

## Citation
```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}

```

```bibtex
@article{Klinker_2010,
	doi = {10.1007/s00591-010-0080-8},
	url = {https://doi.org/10.1007%2Fs00591-010-0080-8},
	year = 2010,
	month = {dec},
	publisher = {Springer Science and Business Media {LLC}
},
	volume = {58},
	number = {1},
	pages = {97--107},
	author = {Frank Klinker},
	title = {Exponential moving average versus moving~exponential~average},
	journal = {Mathematische Semesterberichte}
}
```
