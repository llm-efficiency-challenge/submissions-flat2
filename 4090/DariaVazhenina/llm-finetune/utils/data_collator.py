from torch.nn.utils.rnn import pad_sequence
import sys

class InstructCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        # Create attention_mask with padding token or not
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }