import logging
import typing

try:
    import torch
    from peft import (
        PeftModel,
        __version__ as peft_version,
    )
    from trl import __version__ as trl_version
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        __version__ as transformers_version,
    )
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo.config import TokenizerConfig, InferenceConfig
from cajajejo.model.utils import get_tokenizer, requires_extras
from cajajejo.utils import load_config


class NeuripsPredictor(object):
    def __init__(
        self,
        model: str,
        load_in_8bit: bool,
        torch_dtype: str,
        tokenizer_config: TokenizerConfig,
    ):
        self._logger = logging.getLogger("cajajejo.models.inference.NeuripsPredictor")
        self._logger.debug(f"torch version == {torch.__version__}")
        self._logger.debug(f"transformers version == {transformers_version}")
        self._logger.debug("bitsandbytes version == 0.41.1")
        self._logger.debug(f"trl version == {trl_version}")
        self._logger.debug(f"peft version == {peft_version}")
        self._logger.debug(f"Using model {model}.")
        self._model = model
        self._load_in_8bit = load_in_8bit
        self._torch_dtype = torch_dtype
        self._tokenizer_config = tokenizer_config

    @requires_extras(_has_training_extras, "training")
    def get_trained_lora_model(self, adapter_path: typing.Optional[str] = None):
        model = self.get_model()
        self._logger.info(f"Loading PEFT model from {adapter_path}.")
        peft_model = PeftModel.from_pretrained(
            model,
            adapter_path,
            device_map={"": 0},
            torch_dtype=self._torch_dtype,
        )
        self._logger.info("Merging LoRA weights.")
        return peft_model.merge_and_unload()

    @requires_extras(_has_training_extras, "training")
    def get_model(self):
        bnb_cnf = BitsAndBytesConfig(
            load_in_8bit=self._load_in_8bit,
            torch_dtype=self._torch_dtype,
        )
        self._logger.info(f"Loading model {self._model}")
        model = AutoModelForCausalLM.from_pretrained(
            self._model, quantization_config=bnb_cnf, device_map={"": 0}
        )
        return model

    def get_tokenizer(self):
        return get_tokenizer(
            model=self._model, use_fast=self._tokenizer_config.use_fast
        )

    @classmethod
    def from_config(cls, config_path: str):
        config = load_config(config_path, InferenceConfig)
        return cls(
            model=config.model,
            load_in_8bit=config.load_in_8bit,
            torch_dtype=config.torch_dtype,
            tokenizer_config=config.tokenizer_config,
        )
