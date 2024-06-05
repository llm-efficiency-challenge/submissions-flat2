import json
import pandas as pd
import os
from tqdm import tqdm

def list_dict_to_json(input,output_path):
    df = pd.DataFrame(input)
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    return

file_names = os.listdir('./source_data')
dataset = []
for file in tqdm(file_names):
    file_path = './source_data/'+file
    with open(file_path,'r') as f:
        dataset += json.load(f)

list_dict_to_json(dataset,'../dataset/merged_data.json')
