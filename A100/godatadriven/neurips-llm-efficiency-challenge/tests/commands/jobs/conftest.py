from unittest import mock
import pathlib as plb

import pytest
import yaml

from cajajejo.config import (
    EvaluationConfig,
    HelmConfig,
    ComputeConfig,
    ModelConfig,
    TrackingConfig,
    TrackingConfigTags,
    MlflowArtifactConfig,
)
from cajajejo.commands.jobs.utils import parse_eval_job_and_write_to_file


@pytest.fixture(scope="class", params=["gpu", "none"])
def compute_config(request):
    if request.param == "gpu":
        return ComputeConfig(accelerator="nvidia-tesla-t4", cpu="1", memory="1Gi")
    else:
        return ComputeConfig(accelerator=None, cpu="1", memory="1Gi")


@pytest.fixture(
    scope="class",
    params=[
        ("gs", None, None),
        ("mlflow", "components/tokenizer", None),
        ("mlflow", None, "huggingface/gpt2"),
    ],
)
def model_config(request):
    if request.param[0] == "gs":
        return ModelConfig(
            gs_uri="gs://test-bucket/test-artifact",
        )
    else:
        return ModelConfig(
            mlflow_artifact=MlflowArtifactConfig(
                run_id="5478238igfgyu284t3",
                artifact_path="artifacts",
                model_directory="model",
                tokenizer_directory=request.param[1],
                tokenizer_hf_repo=request.param[2],
            )
        )


@pytest.fixture(scope="class")
def evaluation_config_var(
    compute_config: ComputeConfig,
    model_config: ModelConfig,
    mlflow_tracking_server: plb.Path,
) -> EvaluationConfig:
    return EvaluationConfig(
        version="v6",
        image="test/image",
        image_tag="latest",
        compute_config=compute_config,
        helm=HelmConfig(suite="test-suite", max_eval_instances=10),
        model_config=model_config,
        tracking_config=TrackingConfig(
            tracking_uri=str(mlflow_tracking_server),
            experiment_name="test-experiment",
            description="test-description",
            run_name=None,
            tags=TrackingConfigTags(
                creator="pytest",
                model="test-model",
                datasets=["test-dataset"],
                lifecycle="testing",
            ),
        ),
    )


@pytest.fixture(scope="class")
def evaluation_config_var_on_disk(
    evaluation_config_var,
    tmp_path_factory,
):
    config_path = tmp_path_factory.mktemp("job_configs") / "config.yml"
    with config_path.open("w") as f:
        yaml.dump(evaluation_config_var.dict(), f)
    return str(config_path)


@pytest.fixture(
    scope="class",
    autouse=False,
)
@mock.patch("cajajejo.commands.jobs.utils.generate_slug")
def job_spec(
    mock_coolname_generate_slug,
    evaluation_config_var_on_disk,
    helm_config_on_disk,
    tmp_path_factory,
    request,
):
    mock_coolname_generate_slug.return_value = "test"
    parsed_job_out = tmp_path_factory.getbasetemp() / "parsed_job.yaml"
    parse_eval_job_and_write_to_file(
        path_to_config=str(evaluation_config_var_on_disk),
        path_to_helm_config=helm_config_on_disk,
        job_spec_output_path=str(parsed_job_out),
        config_version="778ff65567df57s6fd",
    )
    with parsed_job_out.open("r") as f:
        parsed_job_spec = yaml.safe_load_all(f.read())
    request.cls.job_spec = [*parsed_job_spec]
