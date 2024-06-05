import pytest

from cajajejo.config import ModelConfig, MlflowArtifactConfig


def test_modelconfig_validation_with_mlflow_run_id():
    ModelConfig(
        mlflow_artifact=MlflowArtifactConfig(
            run_id="5478238igfgyu284t3",
            artifact_path="model",
            model_directory="model",
            tokenizer_directory="components/tokenizer",
        ),
    )


def test_modelconfig_validation_with_gs_uri():
    ModelConfig(
        gs_uri="gs://my-bucket/my-model",
    )


def test_modelconfig_validation_with_both_mlflow_run_id_and_gs_uri():
    with pytest.raises(ValueError):
        ModelConfig(
            mlflow_artifact=MlflowArtifactConfig(
                run_id="5478238igfgyu284t3",
                artifact_path="model",
            ),
            gs_uri="gs://my-bucket/my-model",
        )


# TODO: test modelconfig validator
