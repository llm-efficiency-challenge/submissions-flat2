import pandas as pd
import random
import re
import sys
from datasets import load_dataset

from .base_dataset import BaseDataset

class DollyDataset(BaseDataset):
    def __init__(self, name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort):
        super().__init__(name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort)
        
        # Load texts and labels
        if is_train_or_eval:
            self.dolly = load_dataset("databricks/databricks-dolly-15k")["train"]
        else:
            raise Exception("This dataset don't have test set")
            
        self.instructions = self.dolly["instruction"]
        self.contexts = self.dolly["context"]
        self.labels = self.dolly["response"]
        self.tasks = self.dolly["category"]
    
        # train/val split
        n = 15000
        if split == "train":
            self.instructions = self.instructions[:n]
            self.contexts = self.contexts[:n]
            self.labels = self.labels[:n]
            self.tasks = self.tasks[:n]
        elif split == "val":
            self.instructions = self.instructions[n:]
            self.contexts = self.contexts[n:]
            self.labels = self.labels[n:]
            self.tasks = self.tasks[n:]
            
        if sort == "random":
            self.instructions, self.contexts, self.labels, self.tasks = \
                self._shuffle_at_random(self.instructions, self.contexts, self.labels, self.tasks)
        elif sort == "perplexity":
            self.instructions = self._preprocess(self.instructions)
            self._calculate_perplexity(self.instructions)
            self.instructions, self.contexts, self.labels, self.tasks = \
                self._sort_by_perplexity(self.instructions, self.contexts, self.labels, self.tasks)
            print("Min and max of perplexity:", min(self.perplexities), max(self.perplexities), file=sys.stderr)
        else:
            raise Exception("The sort strategy is not implemented")

        # Get prompts
        self.prompts, self.prompts_with_label = self._generate_prompt()
        
        # If sort strategy is 'perplexity', exclude data with high perplexity by 10%.
        if sort == "perplexity":
            self.prompts = self.prompts[:int(len(self.prompts)*0.9)]
            self.prompts_with_label = self.prompts_with_label[:int(len(self.prompts_with_label)*0.9)]
        
        # Preprocessing
        self.prompts = self._preprocess(self.prompts)
        self.prompts_with_label = self._preprocess(self.prompts_with_label)

    # Overide
    def _generate_prompt(self):
        prompts = []
        prompts_with_label = []
        for idx, (c, i, l, t) in enumerate(zip(self.contexts, self.instructions, self.labels, self.tasks)):
            if t == "brainstorming":
                continue  # No related task in HELM
            elif t == "classification":
                # The classification task in HELM is a RAFT dataset, but the nature of the task is very different 
                # because it is a classification of whether the text is ADE related or not
                continue  # No related task in HELM
            elif t == "closed_qa":
                prompt = "Context: " + c + " " + "Question: " + i + " " + "Answer: "
                prompt_with_label = "Context: " + c + " " + "Question: " + i + " " + "Answer: " + l + self.eos_token
            elif t == "creative_writing": 
                continue  # No related task in HELM
            elif t == "general_qa":
                # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
                prompt = "Question: " + i + " " + "Answer: "
                prompt_with_label = "Question: " + i + " " + "Answer: " + l + self.eos_token
            elif t == "information_extraction":
                # Since MS MARCO is a two-class classification task, I decided on the prompt as a QA task with context.
                # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
                prompt = "Context: " + c + " " + "Question: " + i + " " + "Answer: "
                prompt_with_label = "Context: " + c + " " + "Question: " + i + " " + "Answer: " + l + self.eos_token
            elif t == "open_qa":
                # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
                prompt = "Question: " + i + " " + "Answer: "
                prompt_with_label = "Question: " + i + " " + "Answer: " + l + self.eos_token
            elif t == "summarization":
                # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/summarization_scenario.py
                prompt = "Summarize the given document." + " " + "Document: " + c + " " + "Summary: " 
                prompt_with_label = "Summarize the given document." + " " + "Document: " + c + " " + "Summary: " + l + self.eos_token
            else:
                continue
            
            prompts.append(prompt)
            prompts_with_label.append(prompt_with_label)
            
        return prompts, prompts_with_label