import re
from datasets import load_dataset

from .base_dataset import BaseDataset

class CNNDailyMailDataset(BaseDataset):
    def __init__(self, name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort):
        super().__init__(name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort)

        cnndm = load_dataset("cnn_dailymail", "3.0.0")
        
        if split == "train":
            self.cnndm = cnndm["train"]
        elif split == "val":
            self.cnndm = cnndm["validation"]
        elif split == "test":
            self.cnndm = cnndm["test"]

        self.texts = self.cnndm["article"][:31200]
        self.labels = self.cnndm["highlights"][:31200]
        
        self.texts, self.labels = self._shuffle_at_random(self.texts, self.labels)

        if sort == "random":
            self.texts, self.labels = self._shuffle_at_random(self.texts, self.labels)
        elif sort == "perplexity":
            self._calculate_perplexity(self.texts)
            self.texts, self.labels = self._sort_by_perplexity(self.texts, self.labels)
            print("Min and max of perplexity:", min(self.perplexities), max(self.perplexities), file=sys.stderr)
        else:
            raise Exception("The sort strategy is not implemented")
        
        # Get prompts
        self.inst = "You are given a summarization task. I will now ask you to read the news article, and please summarize its contents in about 50 words as your response. Given text: "
        self.prompts, self.prompts_with_label = self._generate_prompt()
        self.prompts = self._preprocess(self.prompts)
        self.prompts_with_label = self._preprocess(self.prompts_with_label)