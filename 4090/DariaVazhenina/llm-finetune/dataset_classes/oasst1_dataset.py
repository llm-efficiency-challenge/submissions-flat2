import copy
import pandas as pd
import random
import re
import sys
import torch
from datasets import load_dataset
from tqdm import tqdm

from .base_dataset import BaseDataset

class OASST1Dataset(BaseDataset):
    def __init__(self, name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort):
        super().__init__(name, split, is_train_or_eval, max_seq_length, eos_token, tokenizer, sort)
        
        # Load texts and labels
        if split == "train":
            self.oasst1 = load_dataset("OpenAssistant/oasst1")["train"]
        elif split == "val":
            self.oasst1 = load_dataset("OpenAssistant/oasst1")["validation"]
        else:
            raise Exception("This dataset don't have test set")
        
        df_dataset = pd.DataFrame(columns=["message_id", "parent_id", "role", "text"])
        df_dataset["message_id"] = self.oasst1["message_id"]
        df_dataset["parent_id"] = self.oasst1["parent_id"]
        df_dataset["role"] = self.oasst1["role"]
        df_dataset["text"] = self.oasst1["text"]
        df_dataset["lang"] = self.oasst1["lang"]
        df_dataset = df_dataset[df_dataset["lang"]=="en"]

        df_dataset["text"] = self._preprocess(df_dataset["text"].to_list())
        
        self.df_dataset = df_dataset
        self._make_conv()
        
    def _make_conv(self):
        df_dataset = self.df_dataset
        
        # Look for lines where the message_id is not contained in any other parent_id
        end_ids = df_dataset[~df_dataset['message_id'].isin(df_dataset['parent_id'])]['message_id'].tolist() 
        
        conv_list = []
        for target_id in end_ids:
            conversation = []
            while target_id != None:
                # Search for the parent
                df_target = df_dataset[df_dataset["message_id"]==target_id]
                if len(df_target) != 1:
                    # There is a pattern of English prompts for non-English prompts, such as skip with continue
                    conversation = None
                    break
                conversation.append((df_target["role"].to_list()[0], df_target["text"].to_list()[0]))
                target_id = df_target["parent_id"].to_list()[0]
            if conversation is None:
                continue
            else:
                conversation.reverse()
                conv_list.append(conversation) # Since they are stored in the order of child to parent, reverse at the end.
        self.conv_list = conv_list

    # Overide
    def tokenize(self):
        all_input_ids = []
        all_labels = []
        for conv in tqdm(self.conv_list, total=len(self.conv_list)):
            input_ids = None
            labels = None
            for item in conv:
                result = self.tokenizer(item[1], 
                    return_tensors="pt", 
                    truncation=False,
                    add_special_tokens=False)

                if input_ids is None:
                    input_ids = result["input_ids"][0]
                else:
                    input_ids = torch.cat((input_ids, result["input_ids"][0]), dim=0)

                # Process labels
                tmp_labels = copy.deepcopy(result["input_ids"][0])
                if item[0] == "assistant":
                    pass                          # Do not process labels in the generation part.
                elif item[0] == "prompter":
                    tmp_labels[:-1] = self.mask_token # Mask all prompt parts with mask token
                else:
                    raise ValueError("Invalid role")

                if labels is None:
                    labels = tmp_labels
                else:
                    labels = torch.cat((labels, tmp_labels), dim=0)

            # Truncation or padding
            if len(labels) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
                labels = labels[:self.max_seq_length]
            else:
                input_ids = torch.nn.functional.pad(input_ids, (0, self.max_seq_length - len(input_ids)), value=self.tokenizer.pad_token_id)
                labels = torch.nn.functional.pad(labels, (0, self.max_seq_length - len(labels)), value=self.tokenizer.pad_token_id)
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        all_input_ids  = torch.stack(all_input_ids , dim=0).to(torch.int64)
        all_labels  = torch.stack(all_labels , dim=0).to(torch.int64)
        
        self.encodings = {
            "input_ids": all_input_ids,
            "labels": all_labels,
        }

                                    