import pathlib as plb
import typing
import logging
import os

import yaml
from coolname import generate_slug
from pydantic import BaseModel
from jinja2 import Environment, PackageLoader, select_autoescape


from cajajejo.utils import load_config, check_path_exists
from cajajejo.config import EvaluationConfig, TrainingConfig, InferenceConfig

logger = logging.getLogger("cajajejo.commands.jobs.utils")


def _load_helm_config(path: str):
    """Load raw helm config from disk"""
    _path = check_path_exists(path)
    with _path.open("r") as inFile:
        cnf_raw = inFile.read()
    return cnf_raw


def validate_config(path_to_config: str, config_cls: BaseModel):
    """Validate a YAML configuration file from disk"""
    _path_to_config = plb.Path(path_to_config).resolve()
    if not _path_to_config.exists():
        raise FileNotFoundError(f"Config file at '{path_to_config}' does not exist")
    _ = load_config(_path_to_config, config_cls)
    logger.info("🙏 Config validation passed!")


def parse_eval_job_and_write_to_file(
    path_to_eval_config: str,
    path_to_inference_config: str,
    path_to_helm_config: str,
    job_spec_output_path: str,
    config_version: typing.Optional[str] = None,
):
    """Parse a job template and write the parsed job spec to disk"""
    eval_cnf = load_config(path_to_eval_config, EvaluationConfig)
    inf_cnf = load_config(path_to_inference_config, InferenceConfig)
    helm_cnf = _load_helm_config(path_to_helm_config)
    env = Environment(
        loader=PackageLoader("cajajejo", "templates"),
        autoescape=select_autoescape(),
    )
    # NB: control whitespaces. See <https://ttl255.com/jinja2-tutorial-part-3-whitespace-control/>
    env.trim_blocks = True
    env.lstrip_blocks = True
    env.keep_trailing_newline = True
    template = env.get_template("eval_job.yml.j2")
    rendered = template.render(
        image=f"{eval_cnf.image}:{eval_cnf.image_tag}",
        accelerator=inf_cnf.compute_config.accelerator,
        api_cpu=inf_cnf.compute_config.cpu,
        api_memory=inf_cnf.compute_config.memory,
        eval_cpu=eval_cnf.compute_config.cpu,
        eval_memory=eval_cnf.compute_config.memory,
        eval_config=yaml.safe_dump(eval_cnf.dict()),
        inf_config=yaml.safe_dump(inf_cnf.dict()),
        run_specs=helm_cnf,
        config_version=config_version,
        job_name_suffix=generate_slug(2),
        model_run_id=None
        if inf_cnf.mlflow_artifact_config is None
        else inf_cnf.mlflow_artifact_config.run_id,
        model_artifact_path=None
        if inf_cnf.mlflow_artifact_config is None
        else os.path.join(
            inf_cnf.mlflow_artifact_config.artifact_path,
            inf_cnf.mlflow_artifact_config.model_directory,
        ),
        mlflow_tracking_uri=None
        if inf_cnf.mlflow_artifact_config is None
        else inf_cnf.mlflow_artifact_config.mlflow_tracking_uri,
    )
    with plb.Path(job_spec_output_path).open("w") as f:
        f.write(rendered)
    logger.info("🙏 Successfully parsed job!")


def parse_training_job_and_write_to_file(
    path_to_config: str,
    job_spec_output_path: str,
    config_version: typing.Optional[str] = None,
):
    """Parse a job template and write the parsed job spec to disk"""
    cnf = load_config(path_to_config, TrainingConfig)
    env = Environment(
        loader=PackageLoader("cajajejo", "templates"),
        autoescape=select_autoescape(),
    )
    # NB: control whitespaces. See <https://ttl255.com/jinja2-tutorial-part-3-whitespace-control/>
    env.trim_blocks = True
    env.lstrip_blocks = True
    env.keep_trailing_newline = True
    template = env.get_template("training_job.yml.j2")
    rendered = template.render(
        image=f"{cnf.image}:{cnf.image_tag}",
        accelerator=cnf.compute_config.accelerator,
        cpu=cnf.compute_config.cpu,
        memory=cnf.compute_config.memory,
        config=yaml.safe_dump(cnf.dict()),
        config_version=config_version,
        config_path_git_repo=path_to_config,
        job_name_suffix=generate_slug(2),
        mlflow_tracking_uri=cnf.tracking_config.tracking_uri,
        path_to_data_remote=cnf.training_data_config.path_to_data,
        path_to_data_local=os.path.join(
            "/scratch", cnf.training_data_config.path_to_data.split("/")[-1]
        ),
    )
    with plb.Path(job_spec_output_path).open("w") as f:
        f.write(rendered)
    logger.info("🙏 Successfully parsed job!")
