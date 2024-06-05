from cajajejo.utils import load_config
from cajajejo.config import EvaluationConfig


def test_load_eval_config(evaluation_config_on_disk):
    load_config(evaluation_config_on_disk, EvaluationConfig)
