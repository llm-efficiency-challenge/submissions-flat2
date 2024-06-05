import logging
import time
import pathlib as plb
import os

# import gc

import typer

try:
    from fastapi import FastAPI
    import uvicorn

    from cajajejo.utils import load_config
except ImportError:
    _has_api_extras = False
else:
    _has_api_extras = True

try:
    import torch
    import mlflow
    from transformers import AutoModelForCausalLM
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo.commands.api.api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)
from cajajejo.model.inference.predictor import NeuripsPredictor
from cajajejo.model.inference.utils import download_mlflow_artifact
from cajajejo.model.utils import generate_prompt
from cajajejo.config import InferenceConfig
from cajajejo.commands.api.utils import FastApiPlaceholder, is_multiple_choice
from cajajejo.commands.utils import _huggingface_hub_login


logger = logging.getLogger("cajajejo.commands.api.server")

if _has_api_extras:
    app = FastAPI()
else:
    # We need to define a placeholder app so that the command can be loaded
    # Else we cannot load the CLI without the extras
    app = FastApiPlaceholder()

api_cmd = typer.Typer(
    help="🖥  Commands for running an API server to tokenize and generate text.",
    no_args_is_help=True,
)

LLAMA2_CONTEXT_LENGTH = 4096


@app.get("/config")
async def get_config() -> dict:
    return cnf.dict()


@app.get("/quit")
def quit():
    logger.info("Shutting down API server.")
    # Meh to use this but it's the only way to exit the server and pod
    os._exit(0)


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    # Use regex to check if multiple choice question
    prompt_is_mc = is_multiple_choice(input_data.prompt)
    if prompt_is_mc:
        input = "Choose the best option out of the choices given, and return the letter corresponding to the option you choose."
    else:
        input = ""

    if input_data.prompt.startswith("###\nArticle:"):
        temperature = 0.7
    else:
        temperature = input_data.temperature

    prompt = generate_prompt(
        {"instruction": input_data.prompt, "output": "", "input": input}
    )

    encoded = tokenizer(prompt, return_tensors="pt")

    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )

    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(
            outputs.sequences[0][prompt_length:], skip_special_tokens=True
        )
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []

    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1] :]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:, :, None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(
        gen_sequences.tolist()[0],
        gen_logprobs.tolist()[0],
        zip(top_indices, top_logprobs),
    ):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()

    del encoded
    del outputs
    del log_probs
    del gen_sequences
    del gen_logprobs
    del top_indices
    del top_logprobs

    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()

    resp = ProcessResponse(
        text=output.strip(),
        tokens=generated_tokens,
        logprob=logprob_sum,
        request_time=t,
    )

    print(resp)

    return resp


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    prompt_is_mc = is_multiple_choice(input_data.text)
    if prompt_is_mc:
        input = "Choose the best option out of the choices given, and return the letter corresponding to the option you choose."
    else:
        input = ""

    prompt = generate_prompt(
        {"instruction": input_data.text, "output": "", "input": input}
    )
    encoded = tokenizer(prompt)
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)


@api_cmd.command(
    name="start",
    help="Start the API server.",
    short_help="Start the API server.",
    no_args_is_help=True,
)
def _api_server(
    path_to_config: str = typer.Argument(None, help="Path to a training config file."),
    path_to_adapter: str = typer.Option(None, help="Path to a trained adapter."),
    host: str = typer.Option("127.0.0.1", help="Host to run the API server on."),
    port: int = typer.Option(8080, help="Port to run the API server on."),
    dry_run: bool = typer.Option(
        False,
        help="Run the API server using a base Pythia-31M model for debigging purposes.",
    ),
):
    if not (_has_api_extras and _has_training_extras):
        raise ImportError(
            """🚨 You need to install the `cajajejo[submission,training]` extras to use this command.
            If you're using poetry, use `poetry install -E submission -E training`."""
        )
    # ugly but else we cannot use tokenizer/model in the above endpoints
    global tokenizer
    global model
    global cnf
    cnf = load_config(path_to_config, InferenceConfig)
    if cnf.huggingface_token_secret_name is not None:
        _huggingface_hub_login(cnf.huggingface_token_secret_name)
    if cnf.mlflow_artifact_config is not None:
        mlflow_download_dir = plb.Path(cnf.mlflow_artifact_config.download_directory)
        mlflow_artifact_dir = cnf.mlflow_artifact_config.artifact_path
        mlflow_model_dir = cnf.mlflow_artifact_config.model_directory
        if not (mlflow_download_dir / mlflow_artifact_dir).exists():
            mlflow_download_dir.mkdir(parents=False, exist_ok=True)
            if not (
                mlflow_download_dir / mlflow_artifact_dir / mlflow_model_dir
            ).exists():
                mlflow.set_tracking_uri(cnf.mlflow_artifact_config.mlflow_tracking_uri)
                download_mlflow_artifact(
                    cnf.mlflow_artifact_config.run_id,
                    mlflow_artifact_dir,
                    mlflow_download_dir,
                )
    if path_to_adapter is not None:
        _path_to_adapter = path_to_adapter
    elif cnf.mlflow_artifact_config is not None:
        _path_to_adapter = str(
            mlflow_download_dir / mlflow_artifact_dir / mlflow_model_dir
        )
    elif cnf.path_to_adapter is not None:
        _path_to_adapter = cnf.path_to_adapter
    else:
        _path_to_adapter = None
    torch.set_float32_matmul_precision("high")
    predictor = NeuripsPredictor.from_config(path_to_config)
    tokenizer = predictor.get_tokenizer()
    if dry_run:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-31m")
    else:
        if _path_to_adapter is not None:
            model = predictor.get_trained_lora_model(adapter_path=_path_to_adapter)
        else:
            model = predictor.get_model()
    model.eval()
    uvicorn.run(app, port=port, host=host)
