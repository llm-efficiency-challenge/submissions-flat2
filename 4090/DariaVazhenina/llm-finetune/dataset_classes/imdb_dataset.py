import pandas as pd
import re
from datasets import load_dataset
import sys

from .base_dataset import BaseDataset

class IMDBDataset(BaseDataset):
    def __init__(self, name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort):
        super().__init__(name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort)
        
        # Load texts and labels
        if is_train_or_eval:
            self.imdb = load_dataset("imdb")["train"]
        else:
            self.imdb = load_dataset("imdb")["test"]
            
        self.texts = self.imdb["text"]
        self.texts = self._preprocess(self.texts)
        
        self.labels = self.imdb["label"]
        self.labels = ["Negative" if l == 0 else "Positive" for l in self.labels]
        
        # train/val split
        n = 25000
        if split == "train":
            self.texts = self.texts[:n]
            self.labels = self.labels[:n]
        elif split == "val":
            self.texts = self.texts[n:]
            self.labels = self.labels[n:]
        
        if sort == "random":
            self.texts, self.labels = self._shuffle_at_random(self.texts, self.labels)
        elif sort == "perplexity":
            self._calculate_perplexity(self.texts)
            self.texts, self.labels = self._sort_by_perplexity(self.texts, self.labels)
            print("Min and max of perplexity:", min(self.perplexities), max(self.perplexities), file=sys.stderr)
            
            # Exclude data with high perplexity by 10%.
            self.texts = self.texts[:int(n*0.9)]
            self.labels = self.labels[:int(n*0.9)]
        else:
            raise Exception("The sort strategy is not implemented")
        
        # Get prompts
        self.prompts, self.prompts_with_label = self._generate_prompt()
    
    # Override
    def _generate_prompt(self):
        # HELM prompt: https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/imdb_scenario.py
        prompts = [f"{t} Sentiment: " for t in self.texts]
        prompts_with_label = [f"{t} Sentiment: {self.labels[i]}{self.eos_token}" for i, t in enumerate(self.texts)]
   
        return prompts, prompts_with_label