from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("Rocinante/insturction_merge")
print(dataset)

def list_dict_to_json(input,output_path):
    df = pd.DataFrame(input)
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    return

if not os.path.exists('./dataset/'):
    os.mkdir('./dataset/')

list_dict_to_json(dataset['train'],'./dataset/merged_data.json')
