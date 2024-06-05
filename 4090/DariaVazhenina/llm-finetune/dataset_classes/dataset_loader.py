from .imdb_dataset import IMDBDataset
from .cnndm_dataset import CNNDailyMailDataset
from .dolly_dataset import DollyDataset
from .oasst1_dataset import OASST1Dataset
from .csv_dataset import CSVDataset

def load_dataset(name, split, is_train_or_eval, max_seq_length, tokenizer, sort, **kwargs):
    if name == "imdb":
        my_dataset = IMDBDataset(
            name=name,
            split=split,
            is_train_or_eval=is_train_or_eval,
            max_seq_length=max_seq_length,
            eos_token=tokenizer.eos_token,
            tokenizer=tokenizer,
            sort=sort
        )
        my_dataset.tokenize()
        my_dataset.check_tokens()
        my_dataset.check_seq_length()
        
    elif name == "cnndm":
        my_dataset = CNNDailyMailDataset(
            name=name,
            split=split,
            is_train_or_eval=is_train_or_eval,
            max_seq_length=max_seq_length,
            eos_token=tokenizer.eos_token,
            tokenizer=tokenizer,
            sort=sort
        )
        my_dataset.tokenize()
        my_dataset.check_tokens()
        my_dataset.check_seq_length()

    elif name == "dolly":
        my_dataset = DollyDataset(
            name=name,
            split=split,
            is_train_or_eval=is_train_or_eval,
            max_seq_length=max_seq_length,
            eos_token=tokenizer.eos_token,
            tokenizer=tokenizer,
            sort=sort
        )
        my_dataset.tokenize()
        my_dataset.check_tokens()
        my_dataset.check_seq_length()
        
    elif name == "oasst1":
        my_dataset = OASST1Dataset(
            name=name,
            split=split,
            is_train_or_eval=is_train_or_eval,
            max_seq_length=max_seq_length,
            eos_token=tokenizer.eos_token,
            tokenizer=tokenizer,
            sort=sort
        )
        my_dataset.tokenize()

    elif name == "csv":
        my_dataset = CSVDataset(
            name=name,
            split=split,
            is_train_or_eval=is_train_or_eval,
            max_seq_length=max_seq_length,
            eos_token=tokenizer.eos_token,
            tokenizer=tokenizer,
            sort=sort,
            csv_path=kwargs["csv_path"],
        )
        my_dataset.tokenize()
        my_dataset.check_tokens()
        my_dataset.check_seq_length()

    return my_dataset
    
