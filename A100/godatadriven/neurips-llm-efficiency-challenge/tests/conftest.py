import pytest
import pathlib as plb
import yaml

from cajajejo.config import (
    EvaluationConfig,
    ComputeConfig,
    ModelConfig,
    TrackingConfig,
    TrackingConfigTags,
    HelmConfig,
    MlflowArtifactConfig,
    BitsAndBytesConfig,
    BitsAndBytes4bitConfig,
    TrainArgumentsConfig,
    LoraConfig,
    SftTrainerConfig,
    TokenizerConfig,
    ModelMergerConfig,
    TrainingConfig,
    TrainingDataConfig,
)
from cajajejo.utils import yaml_str_presenter

yaml.add_representer(str, yaml_str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, yaml_str_presenter)


@pytest.fixture(scope="session")
def mlflow_tracking_server(tmp_path_factory: plb.Path) -> plb.Path:
    mlflow_server_path = tmp_path_factory.mktemp("mlflow")  # type: ignore
    return mlflow_server_path


@pytest.fixture(scope="session")
def hf_train_artifacts(tmp_path_factory: plb.Path) -> plb.Path:
    hf_train_artifacts = tmp_path_factory.mktemp("hf_outdir")
    return hf_train_artifacts


@pytest.fixture(scope="session")
def training_config(
    mlflow_tracking_server: plb.Path, hf_train_artifacts: plb.Path
) -> TrainingConfig:
    return TrainingConfig(
        model="facebook/opt-125m",
        version="v1",
        seed=42,
        huggingface_token_secret_name="projects/750004521723/secrets/JASPER-HUGGINGFACE-TOKEN/versions/1",
        training_data_config=TrainingDataConfig(
            path_to_data="gs://path/to/data.csv", test_size=0.05
        ),
        sft_trainer_config=SftTrainerConfig(max_seq_length=264),
        bits_and_bytes_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_config=BitsAndBytes4bitConfig(
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            ),
        ),
        lora_config=LoraConfig(
            target_modules=["q_proj", "k_proj", "v_proj"],
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        tokenizer_config=TokenizerConfig(use_fast=True),
        train_arguments_config=TrainArgumentsConfig(
            output_dir=str(hf_train_artifacts),
            save_stategy="steps",
            save_steps=100,
            save_total_limit=3,
            num_train_epochs=1,
            evaluation_strategy="steps",
            eval_steps=10,
            logging_strategy="steps",
            logging_steps=10,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            per_device_train_batch_size=1,
            eval_accumulation_steps=2,
            learning_rate=0.0004,
            lr_scheduler_type="cosine",
            max_grad_norm=0.003,
            warmum_steps=100,
            optim="adam_paged_8bit",
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            fp16=True,
            tf32=False,
            bf16=False,
            group_by_length=True,
            torch_compile=False,
        ),
        model_merger_config=ModelMergerConfig(
            load_in_8bit=False,
            lora_checkpoint="/path/to/lora",
            torch_dtype="float16",
        ),
        tracking_config=TrackingConfig(
            tracking_uri=str(mlflow_tracking_server),
            experiment_name="test",
            description="This is a test",
            tags=TrackingConfigTags(creator="jasper", lifecycle="testing"),
        ),
    )


@pytest.fixture(scope="session")
def evaluation_config(mlflow_tracking_server: plb.Path) -> EvaluationConfig:
    return EvaluationConfig(
        version="v6",
        image="test/image",
        image_tag="latest",
        compute_config=ComputeConfig(
            accelerator="nvidia-tesla-t4", cpu="1", memory="1Gi"
        ),
        helm=HelmConfig(suite="test-suite", max_eval_instances=10),
        model_config=ModelConfig(
            mlflow_artifact=MlflowArtifactConfig(
                run_id="5478238igfgyu284t3",
                artifact_path="model",
                model_directory="model",
                tokenizer_directory="components/tokenizer",
            ),
        ),
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


@pytest.fixture(scope="session")
def helm_config() -> str:
    return """entries: [{description: "mmlu:subject=philosophy,model=huggingface/gpt2", priority: 1}]\n"""


@pytest.fixture(scope="session")
def helm_run_output_path(tmp_path_factory: plb.Path) -> plb.Path:
    helm_run_output = tmp_path_factory.mktemp("benchmark_outputs")  # type: ignore
    (helm_run_output / "test_runs.txt").touch()
    return helm_run_output


@pytest.fixture(scope="session")
def evaluation_config_on_disk(
    tmp_path_factory: plb.Path, evaluation_config: EvaluationConfig
) -> str:
    config_path = tmp_path_factory.getbasetemp() / "config.yml"
    with config_path.open("w") as f:
        yaml.dump(evaluation_config.dict(), f)
    return str(config_path)


@pytest.fixture(scope="session")
def helm_config_on_disk(tmp_path_factory: plb.Path, helm_config: str) -> str:
    config_path = tmp_path_factory.getbasetemp() / "run_specs.conf"
    with config_path.open("w") as f:
        f.write(helm_config)
    return str(config_path)
