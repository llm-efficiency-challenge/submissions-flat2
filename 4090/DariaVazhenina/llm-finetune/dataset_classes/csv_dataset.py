import pandas as pd
import re
from datasets import load_dataset
import sys

from .base_dataset import BaseDataset

class CSVDataset(BaseDataset):
    def __init__(self, name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort, csv_path):
        super().__init__(name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort)
        
        # Load texts and labels
        print(csv_path, file=sys.stderr)
        df_dataset = pd.read_csv(csv_path)
        self.prompts = df_dataset["prompt"].to_list()
        self.prompts_with_label = df_dataset["prompt_with_label"].to_list()
        self.prompts_with_label = [p + tokenizer.eos_token for p in self.prompts_with_label]

        if sort == "random":
            self.prompts, self.prompts_with_label = self._shuffle_at_random(self.prompts, self.prompts_with_label)
        elif sort == "perplexity":
            self._calculate_perplexity(self.prompts)
            self.prompts, self.prompts_with_label = self._sort_by_perplexity(self.prompts, self.prompts_with_label)
            print("Min and max of perplexity:", min(self.perplexities), max(self.perplexities), file=sys.stderr)
        else:
            raise Exception("The sort strategy is not implemented")
