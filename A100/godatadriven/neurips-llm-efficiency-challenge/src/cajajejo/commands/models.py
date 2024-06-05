import logging

from .utils import evaluate_from_config, finetune_model_from_config

import typer

logger = logging.getLogger("cajajejo.commands.models")


models_cmd = typer.Typer(
    help="🧠 Commands for training & evaluating models",
    no_args_is_help=True,
)


@models_cmd.command(
    name="finetune-model",
    help="✨ Finetune a model using a dataset and training config file.",
    short_help="✨ Finetune a model using a training config file.",
    no_args_is_help=True,
)
def _finetune_model(
    path_to_config: str = typer.Argument(None, help="Path to a training config file."),
    path_to_data: str = typer.Option(
        None,
        help="Path to a dataset. If not passed, then the dataset will be resolved from the configuration file.",
    ),
):
    finetune_model_from_config(path_to_config=path_to_config, path_to_data=path_to_data)


@models_cmd.command(
    name="helm-run",
    short_help="🪖 Custom helm run command. We use this one because we add additional functionality.",
    help="🪖 Custom helm run command. We use this one because we add additional functionality.",
    no_args_is_help=True,
)
def custom_helm_run(
    path_to_config: str = typer.Argument(
        None, help="Path to an evaluation config file."
    ),
    path_to_helm_config: str = typer.Argument(None, help="Path to a helm config file."),
    helm_output_path: str = typer.Argument(
        None, help="Path to which the helm output should be written."
    ),
    job_name: str = typer.Option(
        None,
        help="Name of the job. If not provided, will be generated automatically.",
    ),
    api_url: str = typer.Option(
        "http://localhost:8080",
        help="The API URL to use. Should be started using `cajajejo api-server start`.`",
    ),
    skip_sanity_checks: bool = typer.Option(
        True,
        help="Skip sanity checks?",
    ),
    ignore_helm_errors: bool = typer.Option(
        True,
        help="If set to True, will ignore errors from HELM. This is useful because sometimes evaluations will fail, but we still want to register the results with MLFlow.",
    ),
):
    evaluate_from_config(
        path_to_config=path_to_config,
        path_to_helm_config=path_to_helm_config,
        helm_output_path=helm_output_path,
        job_name=job_name,
        api_url=api_url,
        skip_sanity_checks=skip_sanity_checks,
        ignore_helm_errors=ignore_helm_errors,
    )
