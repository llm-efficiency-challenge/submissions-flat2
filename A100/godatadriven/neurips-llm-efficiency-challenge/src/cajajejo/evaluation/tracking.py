import os
import re
import json
import typing
import logging
import pathlib as plb

try:
    import mlflow
    import pandas as pd
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo.utils import requires_extras, parse_tags
from cajajejo.config import EvaluationConfig
from cajajejo.evaluation.utils import _parse_helm_header, _parse_helm_values
from cajajejo import __version__

logger = logging.getLogger("cajajejo.evaluation.tracking")


def _load_core_statistics(core_stats_path: str):
    """Load core statistics from a JSON file"""
    with plb.Path(core_stats_path).open("r") as f:
        core = json.load(f)
    return core


@requires_extras(_has_training_extras, "training")
def _core_statistics_to_dataframe(core_stats: typing.Dict[str, typing.Any]):
    """Convert core statistics to a pandas DataFrame"""
    dfs_metrics = {}
    for m in core_stats:
        header = _parse_helm_header(m["header"])
        metric = _parse_helm_values(m["rows"][0])
        df = pd.DataFrame([metric], columns=header)
        df.index = [m["title"]]
        dfs_metrics[m["title"]] = df
    return dfs_metrics


@requires_extras(_has_training_extras, "training")
def register_results_with_mlflow(
    config: EvaluationConfig,
    job_name: str,
    helm_run_output_path: str,
    eval_config_path: str,
    helm_config_path: str,
    core_stats_path: str,
    inference_config_path: typing.Optional[str] = None,
):
    """Send HELM evaluation results to MLFlow"""
    mlflow.set_tracking_uri(config.tracking_config.tracking_uri)
    mlflow.set_experiment(config.tracking_config.experiment_name)
    tags = parse_tags(config.tracking_config.tags.dict())
    suite = f"{job_name}-{config.tracking_config.tags.model}-{__version__}"
    with mlflow.start_run(
        run_name=config.tracking_config.run_name,
        tags=tags,
        description=config.tracking_config.description,
    ) as run:
        logger.info(f"🏷 MLFlow run ID: {run.info.run_id}")
        mlflow.set_tag("config_git_sha", os.getenv("CONFIG_VERSION"))
        mlflow.set_tag("config_path_git_repo", os.getenv("CONFIG_PATH_GIT_REPO"))
        mlflow.set_tag(
            "helm_config_path_git_repo", os.getenv("HELM_CONFIG_PATH_GIT_REPO")
        )
        mlflow.set_tag("image_build_cajajejo_version", __version__)
        mlflow.set_tag("accelerator", config.compute_config.accelerator)
        mlflow.set_tag("cpu", config.compute_config.cpu)
        mlflow.set_tag("memory", config.compute_config.memory)
        mlflow.set_tag("image", config.image)
        mlflow.set_tag("image_tag", config.image_tag)
        mlflow.set_tag("job_name", job_name)
        for k, v in config.helm.dict().items():
            mlflow.log_param(f"helm_{k}", str(v))
        mlflow.log_param("suite", suite)
        log_core_statistics_to_mlflow(
            core_stats_path=core_stats_path,
        )
        if inference_config_path is not None:
            mlflow.log_artifact(inference_config_path)
        mlflow.log_artifacts(helm_run_output_path)
        mlflow.log_artifact(eval_config_path)
        mlflow.log_artifact(helm_config_path)


@requires_extras(_has_training_extras, "training")
def log_core_statistics_to_mlflow(core_stats_path):
    """Log core statistics to MLFlow"""
    stats = _load_core_statistics(core_stats_path)
    dfs = _core_statistics_to_dataframe(stats)
    for metric_name, metric in dfs.items():
        for k, v in metric.to_dict(orient="records")[0].items():
            if isinstance(v, str) or v is None:
                continue
            metric_name_parsed = re.sub("[^a-zA-Z0-9 \n\.]", "", k).replace(" ", "_")
            mlflow.log_metric(
                f"{metric_name.lower().replace(' ', '_')}_{metric_name_parsed}", v
            )
