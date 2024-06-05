import json
from datasets import Dataset

with open('../dataset/merged_data.json','r') as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset)
print(dataset)

sources = [item['data_source'] for item in dataset]

from collections import Counter
result = Counter(sources)
print(result)
