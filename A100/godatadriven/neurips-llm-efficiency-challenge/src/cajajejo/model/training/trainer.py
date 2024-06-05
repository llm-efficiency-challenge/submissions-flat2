import logging
import pathlib as plb

try:
    import torch
    from peft import (
        get_peft_model,
        LoraConfig,
        prepare_model_for_kbit_training,
        __version__ as peft_version,
    )
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        __version__ as transformers_version,
        set_seed,
    )
    from trl import SFTTrainer, __version__ as trl_version
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo.config import (
    LoraConfig as YamlLoraConfig,
    BitsAndBytesConfig as YamlBitsAndBytesConfig,
    TrainArgumentsConfig,
    TokenizerConfig,
    TrainingConfig,
    SftTrainerConfig,
)
from cajajejo.utils import load_config, requires_extras
from cajajejo.model.utils import get_tokenizer


class NeuripsTrainer:
    def __init__(
        self,
        model: str,
        seed: int,
        sft_trainer_config: SftTrainerConfig,
        lora_config: YamlLoraConfig,
        bits_and_bytes_config: YamlBitsAndBytesConfig,
        train_arguments_config: TrainArgumentsConfig,
        tokenizer_config: TokenizerConfig,
    ):
        """NeuripsTrainer constructor

        Parameters
        ----------
        model : str
            HF model name
        seed : int
            random seed
        sft_trainer_config : SftTrainerConfig
            SFTTrainer configuration (see cajajejo.config)
        lora_config : YamlLoraConfig
            LoraConfig configuration (see cajajejo.config)
        bits_and_bytes_config : YamlBitsAndBytesConfig
            BitsAndBytesConfig configuration (see cajajejo.config)
        train_arguments_config : TrainArgumentsConfig
            TrainingArguments configuration (see cajajejo.config)
        tokenizer_config : TokenizerConfig
            Tokenizer configuration (see cajajejo.config)
        """
        self._logger = logging.getLogger("cajajejo.training.trainer.NeuripsTrainer")
        self._logger.debug(f"torch version == {torch.__version__}")
        self._logger.debug(f"transformers version == {transformers_version}")
        self._logger.debug("bitsandbytes version == 0.41.1")
        self._logger.debug(f"trl version == {trl_version}")
        self._logger.debug(f"peft version == {peft_version}")
        self._logger.debug(f"Using model {model}.")
        self._model = model
        self._sft_trainer_config = sft_trainer_config
        self._lora_config = lora_config
        self._bits_and_bytes_config = bits_and_bytes_config
        self._train_arguments_config = train_arguments_config
        self._tokenizer_config = tokenizer_config
        self._enable_tf32()
        set_seed(seed)

    @requires_extras(_has_training_extras, "training")
    def _enable_tf32(self):
        """Enable TF32 for training"""
        if self._train_arguments_config.tf32:
            self._logger.info("Enabling TF32 for training.")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @requires_extras(_has_training_extras, "training")
    def get_model(self):
        """Retrieve a base model & prepare for 4bit/8bit training"""
        bnb_cnf = self._parse_bnb_config()
        lora_cnf = self._parse_lora_config()
        self._logger.debug(
            f"Using BitsAndBytesConfig: {self._bits_and_bytes_config.dict()}"
        )
        self._logger.info(f"Loading model {self._model}")
        model = AutoModelForCausalLM.from_pretrained(
            self._model, quantization_config=bnb_cnf, device_map={"": 0}
        )
        self._logger.debug(f"Using lora config {self._lora_config.dict()}")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_cnf)
        self._logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
        return model

    def get_tokenizer(self):
        """Retrieve a tokenizer"""
        return get_tokenizer(
            model=self._model, use_fast=self._tokenizer_config.use_fast
        )

    @requires_extras(_has_training_extras, "training")
    def train_model(self, model, train_dataset, eval_dataset, dataset_text_field: str):
        """Use the TRL SFTTrainer to finetune a model"""
        self._logger.info("Setting up SFTTrainer")
        training_args = self._parse_training_arguments_config()
        tokenizer = self.get_tokenizer()
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field=dataset_text_field,
            max_seq_length=self._sft_trainer_config.max_seq_length,
            args=training_args,
            tokenizer=tokenizer,
        )
        self._logger.info(
            f"Saving model checkpoints in {self._train_arguments_config.output_dir}."
        )
        try:
            trainer.evaluate()
            trainer.train()
        except RuntimeError as e:
            self._logger.error("Training failed.")
            self._logger.error(e)
            raise e
        final_model_path = str(
            plb.Path(self._train_arguments_config.output_dir) / "final_model"
        )
        self._logger.info(f"Saving final model in {final_model_path}.")
        model.save_pretrained(
            final_model_path,
            save_adapter=True,
            save_config=True,
        )
        self._logger.info("Training finished.")

    @requires_extras(_has_training_extras, "training")
    def _parse_bnb_config(self):
        """Utility for parsing BitsAndBytesConfig"""
        if self._bits_and_bytes_config.load_in_8bit:
            self._logger.info("Loading model in 8bit.")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            self._logger.info("Loading model in 4bit.")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self._bits_and_bytes_config.bnb_4bit_config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self._bits_and_bytes_config.bnb_4bit_config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=self._bits_and_bytes_config.bnb_4bit_config.bnb_4bit_compute_dtype,
            )

    def _parse_lora_config(self):
        """Utility for parsing LoraConfig"""
        return LoraConfig(
            **self._lora_config.dict(),
        )

    def _parse_training_arguments_config(self):
        """Utility for parsing TrainingArgumentsConfig"""
        return TrainingArguments(
            **self._train_arguments_config.dict(),
        )

    @classmethod
    def from_config(cls, config_path: str):
        """Instantiate a NeuripsTrainer from a YAML training config file"""
        config = load_config(config_path, TrainingConfig)
        return cls(
            model=config.model,
            seed=config.seed,
            lora_config=config.lora_config,
            sft_trainer_config=config.sft_trainer_config,
            bits_and_bytes_config=config.bits_and_bytes_config,
            train_arguments_config=config.train_arguments_config,
            tokenizer_config=config.tokenizer_config,
        )
