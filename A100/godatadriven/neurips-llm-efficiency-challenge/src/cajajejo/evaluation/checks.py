import pathlib as plb
import os
import logging
import urllib.parse

from pydantic import ValidationError

from cajajejo.utils import load_config
from cajajejo.config import EvaluationConfig
from cajajejo.utils import check_host, ConnectionError

logger = logging.getLogger("cajajejo.evaluation.checks")


def _check_eval_config_exists(eval_conf_path: str):
    """Check if the evaluation config file exists."""
    eval_config_plb = plb.Path(eval_conf_path)
    if eval_config_plb.exists():
        logger.info("✅ Sanity check - evaluation config file exists : PASSED!")
    else:
        msg = "❌ Sanity check - evaluation config file exists : FAILED!"
        logger.error(msg)
        raise FileNotFoundError(msg)


def _check_env_var_is_set(key: str):
    """Check if an environment variable is set."""
    if key in os.environ:
        logger.info(f"✅ Sanity check - {key} is set : PASSED!")
    else:
        msg = f"❌ Sanity check - {key} is set : FAILED!"
        logger.error(msg)
        raise KeyError(msg)


def _check_helm_config_exists(helm_conf_path: str):
    """Check if the helm config file exists."""
    helm_config_plb = plb.Path(helm_conf_path)
    if helm_config_plb.exists():
        logger.info("✅ Sanity check - helm config file exists : PASSED!")
    else:
        msg = "❌ Sanity check - helm config file exists : FAILED!"
        logger.error(msg)
        raise FileNotFoundError(msg)


def _check_model_from_mlflow_has_been_downloaded(
    artifact_name: str, base_path: str = "/scratch"
):
    """Check if the helm config file exists."""
    model_path = plb.Path(base_path) / artifact_name
    if model_path.exists():
        logger.info(
            f"✅ Sanity check - model='{artifact_name}' has been downloaded from MLFlow : PASSED!"
        )
    else:
        msg = f"❌ Sanity check - model='{artifact_name}' has been downloaded from MLFlow : FAILED!"
        logger.error(msg)
        raise FileNotFoundError(msg)


def _check_mlflow_model_path_exists(
    artifact_path: str, model_dir: str, base_path: str = "/scratch"
):
    """Check if the model path exists."""
    model_path_plb = plb.Path(base_path) / artifact_path / model_dir
    if model_path_plb.exists():
        logger.info(f"✅ Sanity check - model path='{model_dir}' exists : PASSED!")
    else:
        msg = f"❌ Sanity check - model path='{model_dir}' exists : FAILED!"
        logger.error(msg)
        raise FileNotFoundError(msg)


def _check_validate_and_load_eval_config(eval_conf_path: str) -> EvaluationConfig:
    """Check if the evaluation config file can be loaded."""
    try:
        cnf = load_config(eval_conf_path, EvaluationConfig)
        logger.info("✅ Sanity check - evaluation config can be loaded : PASSED!")
    except ValidationError as e:
        logger.info("❌ Sanity check - evaluation config can be loaded : FAILED!")
        raise e
    return cnf


def _check_connection_to_mlflow_ok(mlflow_tracking_uri: str):
    """Check if we can connect to the MLFlow tracking uri."""
    mlflow_url_parsed = urllib.parse.urlparse(mlflow_tracking_uri)
    if bool(mlflow_url_parsed.scheme):
        status = check_host(mlflow_url_parsed.hostname, mlflow_url_parsed.port)
        if status == 0:
            logger.info(
                f"✅ Sanity check - can connect to MLFlow tracking url on host={mlflow_url_parsed.hostname} and port={mlflow_url_parsed.port} : PASSED!"
            )
        else:
            msg = f"❌ Sanity check - can connect to MLFlow tracking url on host={mlflow_url_parsed.hostname} and port={mlflow_url_parsed.port} : FAILED!"
            logger.error(msg)
            raise ConnectionError(msg)
    else:
        logger.info(
            f"🚸 Sanity check - can connect to MLFlow: a path='{mlflow_tracking_uri}' was used for the tracking server. This means we are not using a remote MLFlow tracking server. Proceed with caution."
        )


def sanity_checks(eval_conf_path: str, helm_conf_path: str):
    """Perform sanity checks before running the evaluation."""
    _check_env_var_is_set("CONFIG_VERSION")
    _check_env_var_is_set("CONFIG_PATH_GIT_REPO")
    _check_env_var_is_set("HELM_CONFIG_PATH_GIT_REPO")
    _check_env_var_is_set("JOB_NAME")
    _check_eval_config_exists(eval_conf_path)
    _check_helm_config_exists(helm_conf_path)
    cnf = _check_validate_and_load_eval_config(eval_conf_path)
    if cnf.model_config.mlflow_artifact is not None:
        _check_model_from_mlflow_has_been_downloaded(
            artifact_name=cnf.model_config.mlflow_artifact.artifact_path
        )
        _check_mlflow_model_path_exists(
            artifact_path=cnf.model_config.mlflow_artifact.artifact_path,
            model_dir=cnf.model_config.mlflow_artifact.model_directory,
        )
    # Can load helm config **
    _check_connection_to_mlflow_ok(cnf.mlflow_config.tracking_uri)
    # _check_helm_run_debug(debug_helm_conf_path, cnf)
