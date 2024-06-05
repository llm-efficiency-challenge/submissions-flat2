import typing
import warnings
import pathlib as plb

from pydantic import BaseModel, validator, root_validator

EVAL_CONFIG_ACCEPTED_VERSIONS = ["v6"]
TRAIN_CONFIG_ACCEPTED_VERSIONS = ["v2"]
INFERENCE_CONFIG_ACCEPTED_VERSIONS = ["v1"]


class DataVersionConfig(BaseModel):
    hash: str
    gs_uri: str
    slug: str
    created: str


class DataConfig(BaseModel):
    remote: str
    local_data_path: str

    @root_validator
    @classmethod
    def check_remote(cls, values) -> dict:
        if not values["remote"].startswith("gs://"):
            raise ValueError(
                f"Invalid remote '{values['remote']}'. Must be a GCS path."
            )
        return values

    @root_validator
    @classmethod
    def check_local_data_path(cls, values) -> dict:
        _path = plb.Path(values["local_data_path"])
        if not _path.exists():
            raise FileNotFoundError(f"Path '{_path}' does not exist.")
        return values


class MlflowArtifactConfig(BaseModel):
    """
    run_id [str]: MLFlow run ID of the model that you want to use
    artifact_path [str]: MLFlow artifact path of the model that you want to use (usually something like 'model')
    model_directory [str]: Directory inside the artifact path where the model is stored (usually something like 'model')
    tokenizer_directory Optional[str]: Directory inside the artifact path where the tokenizer is stored (usually something like 'components/tokenizer')
    tokenizer_hf_repo Optional[str]: HuggingFace repository name of the tokenizer that you want to use (e.g. 'upstage/llama-30b-instruct')
    """

    mlflow_tracking_uri: str
    run_id: str
    artifact_path: str
    model_directory: str
    download_directory: typing.Optional[str] = None


class ModelConfig(BaseModel):
    """
    mlflow_artifact [MlflowArtifactConfig]: MLFlow artifact config
    gs_uri [str]: Google Cloud Storage URI of the model that you want to use (folder level)
    """

    mlflow_artifact: typing.Optional[MlflowArtifactConfig] = None
    gs_uri: typing.Optional[str] = None

    @root_validator
    @classmethod
    def model_config_validation(cls, values: typing.Dict) -> typing.Dict:
        if values["mlflow_artifact"] is None and values["gs_uri"] is None:
            raise ValueError(
                "Either 'mlflow_artifact' or 'gs_uri' must be specified in model config."
            )
        if values["mlflow_artifact"] is not None and values["gs_uri"] is not None:
            raise ValueError(
                "Only one of 'mlflow_artifact' or 'gs_uri' can be specified in model config."
            )
        return values


class TrainingDataConfig(BaseModel):
    path_to_data: str
    test_size: float = 0.05

    @validator("test_size")
    @classmethod
    def check_test_size(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"Invalid test_size '{v}'. Must be between 0 and 1.")
        return v

    @validator("path_to_data")
    @classmethod
    def check_path_to_data(cls, v: str) -> str:
        if not v.startswith("gs://"):
            raise ValueError(f"Invalid path_to_data '{v}'. Must be a GCS path.")
        return v


class TrackingConfigTags(BaseModel):
    """
    creator [str]: Name of the creator of the MLFlow run
    model [str]: Name of the model
    datasets [List[str]]: List of datasets used to train the model
    lifecycle [str]: Lifecycle stage of the model (e.g. 'production', 'experimentation')
    """

    creator: str
    model: typing.Optional[str] = None
    datasets: typing.Optional[typing.List[str]] = None  # Not relevant for eval jobs
    lifecycle: str


class TrackingConfig(BaseModel):
    """
    experiment_name [str]: Name of the MLFlow experiment
    description [str]: Description of the MLFlow experiment
    tags [TrackingConfigTags]: Tags for the MLFlow experiment
    run_name Optional[str]: Name of the MLFlow run
    """

    tracking_uri: str
    experiment_name: str
    description: str
    tags: TrackingConfigTags
    run_name: typing.Optional[str] = None


class ComputeConfig(BaseModel):
    """
    accelerator Optional[str]: Accelerator to use for the model (e.g. 'nvidia-tesla-t4', 'nvidia-tesla-a100', 'nvidia-a100-80gb')
    cpu [str]: Number of CPUs to use for the model (e.g. '1', '2', '4')
    memory [str]: Amount of memory to use for the model (e.g. '1Gi', '2Gi', '4Gi')
    """

    accelerator: typing.Optional[str] = None
    cpu: str
    memory: str

    @validator("accelerator")
    @classmethod
    def check_valid_accelerator(cls, v: str) -> str:
        if v is not None:
            if v not in ["nvidia-tesla-t4", "nvidia-tesla-a100", "nvidia-a100-80gb"]:
                raise ValueError(
                    f"Invalid accelerator '{v}'. Must be one of 'nvidia-tesla-t4', 'nvidia-tesla-a100', 'nvidia-a100-80gb'."
                )
        return v


class HelmConfig(BaseModel):
    """
    args [List[str]]: List of helm arguments
    suite [str]: Name of your helm evaluation suite (this can be anything)
    """

    max_eval_instances: typing.Optional[int] = 100

    # Should not touch these as they are set in k8s cluster
    # output_path: typing.Optional[str] = "/scratch/benchmark_output"
    # conf_paths: typing.Optional[typing.List[str]] = ["/etc/configs/run_specs.conf"]
    suite: typing.Optional[str] = None

    # Configurable
    api_key_path: typing.Optional[str] = "proxy_api_key.txt"
    models_to_run: typing.Optional[typing.List[str]] = None
    groups_to_run: typing.Optional[typing.List[str]] = None
    exit_on_error: typing.Optional[bool] = None
    skip_completed_runs: typing.Optional[bool] = None
    priority: typing.Optional[int] = None
    run_specs: typing.Optional[typing.List[str]] = []
    enable_huggingface_models: typing.Optional[typing.List[str]] = []
    enable_local_huggingface_models: typing.Optional[typing.List[str]] = []
    num_threads: typing.Optional[int] = 4
    skip_instances: typing.Optional[bool] = False
    dry_run: typing.Optional[bool] = False
    num_train_trials: typing.Optional[int] = None
    local_path: typing.Optional[str] = "prod_env"


class TrainArgumentsConfig(BaseModel):
    output_dir: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    num_train_epochs: int
    evaluation_strategy: str
    eval_steps: int
    logging_strategy: str
    logging_steps: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    per_device_eval_batch_size: int
    eval_accumulation_steps: int
    learning_rate: float
    lr_scheduler_type: str
    max_grad_norm: float
    warmup_steps: int
    optim: str
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 1
    fp16: typing.Optional[bool] = False
    tf32: typing.Optional[bool] = False
    bf16: typing.Optional[bool] = False
    group_by_length: bool
    torch_compile: bool = False
    max_steps: typing.Optional[int] = -1

    @root_validator
    @classmethod
    def train_args_config_validation(cls, values: typing.Dict) -> typing.Dict:
        if values["fp16"] and (values["tf32"] or values["bf16"]):
            raise ValueError(
                "Only one of 'fp16', 'tf32' or 'bf16' can be specified in train arguments config."
            )
        return values


class LoraConfig(BaseModel):
    target_modules: typing.List[str]
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str


class BitsAndBytes4bitConfig(BaseModel):
    bnb_4bit_use_double_quant: typing.Optional[bool] = None
    bnb_4bit_quant_type: typing.Optional[str] = None
    bnb_4bit_compute_dtype: typing.Optional[str] = None


class BitsAndBytesConfig(BaseModel):
    load_in_4bit: typing.Optional[bool] = False
    load_in_8bit: typing.Optional[bool] = False
    bnb_4bit_config: typing.Optional[BitsAndBytes4bitConfig] = None

    @root_validator
    @classmethod
    def bnb_config_validation(cls, values: typing.Dict) -> typing.Dict:
        if values["load_in_4bit"] and values["load_in_8bit"]:
            raise ValueError(
                "Only one of 'load_in_4bit' or 'load_in_8bit' can be specified in model config."
            )
        if values["load_in_8bit"] and values["bnb_4bit_config"] is not None:
            warnings.warn(
                "You set 'load_in_8bit' to True but also passed a 4bit configuration. The latter will be ignored."
            )
        return values


class TokenizerConfig(BaseModel):
    use_fast: bool = False


class SftTrainerConfig(BaseModel):
    max_seq_length: int


class TrainingConfig(BaseModel):
    version: str
    image: str
    image_tag: str
    model: str
    seed: int
    huggingface_token_secret_name: typing.Optional[str] = None
    training_data_config: TrainingDataConfig
    compute_config: ComputeConfig
    sft_trainer_config: SftTrainerConfig
    bits_and_bytes_config: BitsAndBytesConfig
    lora_config: LoraConfig
    train_arguments_config: TrainArgumentsConfig
    tokenizer_config: TokenizerConfig
    tracking_config: TrackingConfig

    @validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        if v not in TRAIN_CONFIG_ACCEPTED_VERSIONS:
            raise ValueError(
                f"Invalid version '{v}'. Must be one of {TRAIN_CONFIG_ACCEPTED_VERSIONS}."
            )
        return v


class InferenceConfig(BaseModel):
    version: str
    image: str
    image_tag: str
    model: str
    load_in_8bit: bool
    torch_dtype: str
    tokenizer_config: TokenizerConfig
    path_to_adapter: typing.Optional[str] = None
    mlflow_artifact_config: typing.Optional[MlflowArtifactConfig] = None
    compute_config: ComputeConfig
    huggingface_token_secret_name: typing.Optional[str] = None

    @validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        if v not in INFERENCE_CONFIG_ACCEPTED_VERSIONS:
            raise ValueError(
                f"Invalid version '{v}'. Must be one of {INFERENCE_CONFIG_ACCEPTED_VERSIONS}."
            )
        return v

    @root_validator
    @classmethod
    def check_artifact_path(cls, values: typing.Dict) -> typing.Dict:
        if (
            values["mlflow_artifact_config"] is not None
            and values["path_to_adapter"] is not None
        ):
            raise ValueError(
                "Only one of 'mlflow_artifact_config' or 'path_to_adapter' can be specified in inference config."
            )
        return values


class EvaluationConfig(BaseModel):
    """
    version [str]: Version of the config (e.g. 'v1', 'v2', 'v3')
    image [str]: Name of the image to use for the model
    image_tag [str]: Tag of the image to use for the model (e.g. 'latest' but I recommend using a git commit hash)
    compute_config [ComputeConfig]: Compute config
    helm [HelmConfig]: Helm config
    mlflow_config [MlflowConfig]: MLFlow config
    model_config [ModelConfig]: Model config
    tracking_config [TrackingConfig]: Tracking config
    """

    version: str
    image: str
    image_tag: str
    compute_config: ComputeConfig
    helm: HelmConfig
    tracking_config: TrackingConfig

    @validator("version")
    @classmethod
    def check_version(cls, v: str) -> str:
        if v not in EVAL_CONFIG_ACCEPTED_VERSIONS:
            raise ValueError(
                f"Invalid version '{v}'. Must be one of {EVAL_CONFIG_ACCEPTED_VERSIONS}."
            )
        return v
