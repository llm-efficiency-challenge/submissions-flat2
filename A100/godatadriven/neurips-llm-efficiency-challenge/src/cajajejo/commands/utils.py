import pathlib as plb
import typing
import logging
import os
import urllib
import tempfile
import time
from contextlib import contextmanager

import yaml
import requests
import sh
from coolname import generate_slug
from pydantic import BaseModel

try:
    import pandas as pd
    from datasets import Dataset
    import mlflow
    from huggingface_hub._login import _login
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

try:
    from helm.benchmark.presentation.run_entry import RunEntry, read_run_entries

    # from helm.common.hierarchical_logger import hlog, htrack, htrack_block
    from helm.common.authentication import Authentication
    from helm.proxy.clients.huggingface_model_registry import (
        register_huggingface_hub_model_config,
        register_huggingface_local_model_config,
    )
    from helm.proxy.clients.remote_model_registry import check_and_register_remote_model
    from helm.proxy.services.remote_service import (
        create_authentication,
    )

    # from helm.benchmark.model_metadata_registry import register_model_metadata_from_path
    # from helm.benchmark.model_deployment_registry import (
    #    register_model_deployments_from_path,
    # )

    from helm.benchmark.run import run_benchmarking, run_entries_to_run_specs
    from helm.benchmark.runner import RunnerError
except ImportError:
    _helm_installed = False
else:
    _helm_installed = True

from cajajejo.utils import (
    load_config,
    requires_extras,
    PandasDataFrame,
    HuggingFaceDataset,
    check_host,
    ConnectionError,
    check_path_exists,
    get_secret,
)
from cajajejo.config import EvaluationConfig, TrainingConfig, InferenceConfig
from cajajejo.model.utils import generate_prompt
from cajajejo.model.training.trainer import NeuripsTrainer
from cajajejo.model.training.tracking import neurips_mlflow_tracking
from cajajejo.evaluation.checks import sanity_checks
from cajajejo.evaluation.utils import (
    run_sh_with_exception_handling,
)
from cajajejo.evaluation.tracking import register_results_with_mlflow
from cajajejo import __version__

logger = logging.getLogger("cajajejo.commands.utils")


@requires_extras(_has_training_extras, "training")
def _huggingface_hub_login(huggingface_token_secret_name: str):
    logging.debug(f"Retrieving value for secret {huggingface_token_secret_name}")
    secret = get_secret(huggingface_token_secret_name)
    logging.info("🔐 Logging into HuggingFace Hub")
    _login(token=secret, add_to_git_credential=False)


@requires_extras(_has_training_extras, "training")
def _load_dataset(path: str) -> PandasDataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.split('.')[-1]}")
    return df


@requires_extras(_has_training_extras, "training")
def _preprocess_dataset(df: PandasDataFrame) -> PandasDataFrame:
    df["prompt"] = df.apply(generate_prompt, axis=1)
    df = df[["prompt", "output"]]
    df["text"] = df["prompt"] + df["output"]
    df.drop(columns=["prompt", "output"], inplace=True)
    return df


@requires_extras(_has_training_extras, "training")
def _load_and_preprocess_dataset(
    path: str, seed: int, test_size: float = 0.05
) -> HuggingFaceDataset:
    df = _load_dataset(path)
    df = _preprocess_dataset(df)
    dataset = Dataset.from_pandas(df)
    return dataset.train_test_split(test_size=test_size, seed=seed)


@requires_extras(_has_training_extras, "training")
def finetune_model_from_config(
    path_to_config: typing.Union[str, plb.Path],
    path_to_data: typing.Optional[str] = None,
):
    config_path = check_path_exists(path_to_config)
    cnf = load_config(path_to_config, TrainingConfig)
    if path_to_data is None:
        _path_to_data = cnf.training_data_config.path_to_data
    else:
        _path_to_data = path_to_data
    if cnf.huggingface_token_secret_name is not None:
        _huggingface_hub_login(cnf.huggingface_token_secret_name)
    test_size = cnf.training_data_config.test_size
    seed = cnf.seed
    dataset_path = check_path_exists(_path_to_data)
    dataset = _load_and_preprocess_dataset(str(dataset_path), seed, test_size)
    with neurips_mlflow_tracking(config_path):
        mds_train = mlflow.data.huggingface_dataset.from_huggingface(dataset["train"])
        mds_test = mlflow.data.huggingface_dataset.from_huggingface(dataset["test"])
        mlflow.log_input(mds_train, context="training")
        mlflow.log_input(mds_test, context="evaluation")
        trainer = NeuripsTrainer.from_config(str(config_path))
        model = trainer.get_model()
        trainer.train_model(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            dataset_text_field="text",
        )


class HelmRuntimeError(Exception):
    pass


class ApiKeyPath(BaseModel):
    """Need this because of a HELM function that accesses an attribute"""

    api_key_path: typing.Optional[str] = None


@contextmanager
def quit_api_server_on_exit(api_url):
    try:
        yield
    finally:
        try:
            resp = requests.get(f"{api_url}/quit")
            logger.debug(f"Response: {resp.json()}")
        except ConnectionError:
            logger.info("Server has shut down. Exiting now ... ")


def evaluate_from_config(
    path_to_config: str,
    path_to_helm_config: str,
    helm_output_path: str,
    job_name: typing.Optional[str] = None,
    api_url: str = "http://localhost:8080",
    skip_sanity_checks: bool = True,
    ignore_helm_errors: bool = True,
    api_timeout_seconds: int = 600,  # 10 min
):
    with quit_api_server_on_exit(api_url):
        host_parsed = urllib.parse.urlparse(api_url)
        total_polled = 0
        while total_polled < api_timeout_seconds:
            status = check_host(host_parsed.hostname, host_parsed.port)
            if status != 0:
                logger.info(
                    f"Could not connect to API on host={host_parsed.hostname} and port={host_parsed.port}."
                    " Retrying in 10 seconds."
                )
                total_polled += 10
                time.sleep(5)
            else:
                break
        if status != 0:
            raise ConnectionError(
                f"Could not connect to API on host={host_parsed.hostname} and port={host_parsed.port}."
                " Please check that the API is running. If you're running the API locally, please first"
                " ensure that you have installed cajajejo with 'training' and 'submission' extras and"
                " execute 'cajajejo api-server start' in a separate terminal. You can also increase the"
                " timeout by setting the 'api_timeout_seconds' parameter in case you are loading a large"
                " model."
            )
        else:
            logger.info(f"🔗 Connected to API after {total_polled} seconds.")
        if not _helm_installed:
            raise ImportError(
                "The 'helm' package is not installed. "
                "Please install it using 'pip install helm' or 'poetry run pip install helm'. "
                "Then, reinstall the cajajejo library with the 'training' extras."
            )
        if not skip_sanity_checks:
            sanity_checks(path_to_config, path_to_helm_config)
        if job_name is None:
            _job_name = generate_slug(2)
        else:
            _job_name = job_name
        config = load_config(path_to_config, EvaluationConfig)
        logger.info("🏗 Starting evaluation job ...")
        # Get inference config from API
        resp = requests.get(f"{api_url}/config")
        inference_config = InferenceConfig(**resp.json())
        if inference_config.mlflow_artifact_config is None:
            _model_name = f"{inference_config.model}-base"
        else:
            _model_name = f"{inference_config.model}-ft"
        suite = f"{_job_name}-{_model_name}-{__version__}"
        try:
            _helm_cmd(
                enable_huggingface_models=config.helm.enable_huggingface_models,
                enable_local_huggingface_models=config.helm.enable_local_huggingface_models,
                model_metadata_paths=[],
                model_deployment_paths=[],
                server_url=None,
                enable_remote_models=[],
                conf_paths=[path_to_helm_config],
                run_specs=config.helm.run_specs,
                max_eval_instances=config.helm.max_eval_instances,
                num_train_trials=config.helm.num_train_trials,
                models_to_run=config.helm.models_to_run,
                groups_to_run=config.helm.groups_to_run,
                priority=config.helm.priority,
                skip_instances=config.helm.skip_instances,
                local_path=config.helm.local_path,
                num_threads=config.helm.num_threads,
                output_path=helm_output_path,
                suite=suite,
                dry_run=config.helm.dry_run,
                cache_instances=False,
                cache_instances_only=False,
                skip_completed_runs=config.helm.skip_completed_runs,
                exit_on_error=config.helm.exit_on_error,
                mongo_uri="",
                api_key_path=config.helm.api_key_path,
            )
        except HelmRuntimeError as e:
            if not ignore_helm_errors:
                raise e
            else:
                logger.error(
                    "Some runs failed to complete. Will continue with creating summary statistics."
                )
        logger.info("✅ Evaluation job finished!")
        logger.info("📊 Summarizing results ...")
        run_sh_with_exception_handling(
            sh.helm_summarize,
            ["--suite", suite, "--output-path", helm_output_path],
        )
        logger.info("📝 Logging results to MLFlow ...")
        with tempfile.TemporaryDirectory() as tmpdir:
            path_to_inference_config = os.path.join(tmpdir, "inference_config.yml")
            with open(path_to_inference_config, "w") as f:
                yaml.dump(inference_config.dict(), f)
            register_results_with_mlflow(
                config,
                _job_name,
                os.path.join(helm_output_path, "runs"),
                path_to_config,
                path_to_helm_config,
                os.path.join(
                    helm_output_path,
                    "runs",
                    suite,
                    "groups",
                    "core_scenarios.json",
                ),
                path_to_inference_config,
            )
            logger.info("✅ Logging finished! Exiting ...")


def _helm_cmd(
    enable_huggingface_models,
    enable_local_huggingface_models,
    model_metadata_paths,
    model_deployment_paths,
    api_key_path,
    server_url,
    enable_remote_models,
    conf_paths,
    run_specs,
    max_eval_instances,
    num_train_trials,
    models_to_run,
    groups_to_run,
    priority,
    skip_instances,
    local_path,
    num_threads,
    output_path,
    suite,
    dry_run,
    cache_instances,
    cache_instances_only,
    skip_completed_runs,
    exit_on_error,
    mongo_uri,
):
    """
    This replaces / copies the 'helm-run' command from the HELM python library
    Due to issues with running the command from the library (we never see logs if we call the CLI
    using `sh`), we instead call the relevant commands directly.
    """
    for huggingface_model_name in enable_huggingface_models:
        register_huggingface_hub_model_config(huggingface_model_name)
    for huggingface_model_path in enable_local_huggingface_models:
        register_huggingface_local_model_config(huggingface_model_path)
    # for model_metadata_path in model_metadata_paths:
    #    register_model_metadata_from_path(model_metadata_path)
    # for model_deployment_paths in model_deployment_paths:
    #    register_model_deployments_from_path(model_deployment_paths)

    if server_url and enable_remote_models:
        check_and_register_remote_model(server_url, enable_remote_models)

    run_entries: typing.List[RunEntry] = []
    if conf_paths:
        run_entries.extend(read_run_entries(conf_paths).entries)
    if run_specs:
        run_entries.extend(
            [
                RunEntry(description=description, priority=1, groups=None)
                for description in run_specs
            ]
        )

    run_specs = run_entries_to_run_specs(
        run_entries=run_entries,
        max_eval_instances=max_eval_instances,
        num_train_trials=num_train_trials,
        models_to_run=models_to_run,
        groups_to_run=groups_to_run,
        priority=priority,
    )
    logger.info(f"{len(run_entries)} entries produced {len(run_specs)} run specs")

    if len(run_specs) == 0:
        logger.info("There were no RunSpecs or they got filtered out.")
        return

    auth: Authentication = (
        Authentication("")
        if skip_instances or not server_url
        else create_authentication(ApiKeyPath(api_key_path=api_key_path))
    )

    try:
        run_benchmarking(
            run_specs=run_specs,
            auth=auth,
            url=server_url,
            local_path=local_path,
            num_threads=num_threads,
            output_path=output_path,
            suite=suite,
            dry_run=dry_run,
            skip_instances=skip_instances,
            cache_instances=cache_instances,
            cache_instances_only=cache_instances_only,
            skip_completed_runs=skip_completed_runs,
            exit_on_error=exit_on_error,
            mongo_uri=mongo_uri,
            runner_class_name=None,
        )
    except RunnerError as e:
        logger.error(e)
        raise HelmRuntimeError(
            str(e)
        )  # Using this so we can catch exception in CLI without having to import HELM

    logger.info("Done.")
