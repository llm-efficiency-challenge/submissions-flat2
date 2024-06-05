import copy
import numpy as np
import pandas as pd
import random
import re
import sys
import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort):
        self.split = split
        self.is_train_or_eval = is_train_or_eval
        self.max_seq_length = max_seq_length
        self.eos_token = eos_token
        self.mask_token_id =-100
        self.tokenizer = tokenizer
        self.sort = sort
        
        if self.sort == "perplexity":
            self.df_corpus = pd.read_csv(f"./data/{name}_corpus.csv").set_index("vocab_id")
    
    def _preprocess(self, texts):
        pattern_remained_chars = r'[^A-Za-z0-9.,!?:;\'"“”\-\[\]{}()…_$%&+\s]'
        pattern_symbol_seq     = r"([,!?:;'\-“”\[\]{}()…_])\1{2,}"
        texts = [re.sub('<.*?>', ' ', t) for t in texts]               # Delete html tags
        texts = [re.sub('http\S+', ' ', t) for t in texts]             # Delete links
        texts = [re.sub(pattern_remained_chars, '', t) for t in texts] # Leave only alphanumeric characters, common symbols, any space character
        texts = [re.sub(pattern_symbol_seq, r"\1", t) for t in texts]  # Replace three or more repetitions of the same symbol with a single symbol
        texts = [re.sub('\s+', ' ', t) for t in texts]                 # Replace a sequence of whitespace with a single whitespace character
        texts = [t.strip() for t in texts]                              # strip
    
        return texts
    
    def tokenize(self):
        result = self.tokenizer(
            self.prompts_with_label,
            truncation=True,
            max_length=self.max_seq_length,
            padding="longest",
            return_tensors='pt',
        )

        labels = copy.deepcopy(result["input_ids"])
        
        # Get the lengh of instruction part
        result_inst = self.tokenizer(
            self.prompts,
            truncation=True,
            max_length=self.max_seq_length,
            return_length=True,
        )
        
        # Replace instruction with mask tokens. The part is ignored by the attention_mask in back propagation.
        for i, len_inst in enumerate(result_inst['length']):
            labels[i][:len_inst] = self.mask_token_id
        
        self.encodings = {
            "input_ids": result["input_ids"],
            "labels": labels,
        }

    def _shuffle_at_random(self, *lists):
        # Raises an exception if the lengths of all lists do not match 
        list_lengths = [len(lst) for lst in lists]
        if len(set(list_lengths)) != 1:
            raise ValueError("All input lists must have the same length")

        # Create an index list for shuffle 
        index_list = list(range(list_lengths[0]))
        random.shuffle(index_list)
        
        # Shuffle the list passed as argument 
        shuffled_items = []
        for lst in lists:
            shuffled_list = [lst[i] for i in index_list]
            shuffled_items.append(shuffled_list)

        return tuple(shuffled_items)

    def _calculate_perplexity(self, texts):
        perplexities = []
        for t in texts:
            results = self.tokenizer(
                t,
                truncation=False,
                padding=False,
            )
            sum_log_prob = 0
            # https://huggingface.co/docs/transformers/perplexity
            for id in results["input_ids"]:
                prob = self.df_corpus.loc[id, "prob"]
                log_prob = np.log(prob)
                sum_log_prob += log_prob
            
            if sum_log_prob == 0:
                print("sum_log_prob of given text is 0. Perplexity was set to 0.", file=sys.stderr)
                print("given text:", t, file=sys.stderr)
                perplexity = 0
            else:
                perplexity =  np.exp(-1 / len(results["input_ids"]) * sum_log_prob)
            perplexities.append(perplexity)
            
        self.perplexities = perplexities

    def _sort_by_perplexity(self, *lists):
        # Raises an exception if the lengths of all lists do not match 
        list_lengths = [len(lst) for lst in lists]
        if len(set(list_lengths)) != 1:
            raise ValueError("All input lists must have the same length")
            
        # Get the index of self.perplexities sorted in ascending order
        sorted_indices = np.argsort(self.perplexities)
        sorted_perplexities = [self.perplexities[i] for i in sorted_indices]
        self.perplexities = sorted_perplexities
        
        # Sort items in ascending order of perplexity  
        shuffled_items = []
        for lst in lists:
            shuffled_list = [lst[i] for i in sorted_indices]
            shuffled_items.append(shuffled_list)
            
        return tuple(shuffled_items)
                              
    def check_tokens(self):
        print(
            "prompt:", self.prompts[0], "\n",
            "prompt_with_label:", self.prompts_with_label[0], "\n",
            "input_ids:", self.encodings["input_ids"][0], "\n", 
            "label_ids:", self.encodings["labels"][0] if self.is_train_or_eval else "None"
            , file=sys.stderr)
        
    def check_seq_length(self):
        lengths = [len(t.split(" ")) for t in self.prompts_with_label]
        print("max_seq_length of prompts_with_label:", max(lengths), file=sys.stderr)
        print("ave_seq_length of prompts_with_label:", sum(lengths)/len(lengths), file=sys.stderr)
        
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        
    def __len__(self):
        return len(self.prompts)