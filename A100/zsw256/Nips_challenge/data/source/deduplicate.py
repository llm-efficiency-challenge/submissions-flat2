import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import os


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(text,model,tokenizer):
    with torch.no_grad():
        encoded_input = tokenizer.batch_encode_plus(text, padding=True, truncation=True, return_tensors='pt').to(model.device)
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

def get_prompt(item):
    output=''
    if 'history' in item.keys():
        for turn in item['history']:
            output+='Human:\n'+turn[0]+'\nAssitance:\n'+turn[1]+'\n'
        output+='Human:\n'+item['instruction']+'\nAssitance:\n'
        output+=item['output']
    else:
        if item['input'] is not None:
            output+=item['instruction']+'\n'+item['input']+'\n'+item['output']
        else:
            output=item['instruction']+'\n'+item['output']
    return output

def get_embeddings_dataset(dataset,model,tokenizer,batch_size=32):
    dataset_with_embeddings=[]
    for i in tqdm(range(0, len(dataset), batch_size)):
            step_len = min(len(dataset)-i,batch_size)
            batch_dataset = dataset[i:i+step_len]
            batch = [get_prompt(item) for item in batch_dataset]
            embeddings = embed_text(batch,model,tokenizer)
            embeddings_list = embeddings.split(1, dim=0)
            for i in range(len(embeddings_list)):
                item = batch_dataset[i]
                item['embedding']=embeddings_list[i]
                dataset_with_embeddings.append(item)
    del dataset
    return dataset_with_embeddings

def cosine_similarity(a, b):
    a = torch.nn.functional.normalize(a, p=2, dim=1)
    b = torch.nn.functional.normalize(b, p=2, dim=1)
    return (a * b).sum(dim=1)

def cosine_similarities(a, b_list):
    a = a.unsqueeze(1) # (B, 1, D)
    b = torch.stack(b_list, dim=1) # (B, N, D)
    
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    
    sim = (a_norm * b_norm).sum(dim=-1) #(B, N)
    return sim.tolist()

def list_dict_to_json(input,output_path):
    df = pd.DataFrame(input)
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    return

def deduplicate(data_in,data_out,data_ref):


    device = 'cuda:0'
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name,torch_dtype=torch.bfloat16).eval().to(device)
    with open(data_in,'r') as f:
        dataset = json.load(f)
    if data_ref is not None:
        with open(data_ref,'r') as f:
            dataset_selected = json.load(f)
    else:
        dataset_selected=dataset[:1]

    print('Length of undeduplicated dataset:')
    print(len(dataset))
    print('Length of clean dataset:')
    print(len(dataset_selected))

    dataset = get_embeddings_dataset(dataset,model,tokenizer,batch_size=256)
    dataset_selected = get_embeddings_dataset(dataset_selected,model,tokenizer,batch_size=256)
    list_ref=[ref['embedding'] for ref in dataset_selected]

    for item in tqdm(dataset):
        if max(cosine_similarities(item['embedding'],list_ref)[0])<0.8:
            dataset_selected.append(item)
            list_ref.append(item['embedding'])

    print('Length of merged dataset:')
    print(len(dataset_selected))
    dataset_deduplicatd=[]

    for item in dataset_selected:
        if 'history' in item.keys():
            dataset_deduplicatd.append({
                'instruction':item['instruction'],
                'input':item['input'],
                'output':item['output'],
                'history':item['history'],
                'data_source':item['data_source']
            })
        else:
            dataset_deduplicatd.append({
                'instruction':item['instruction'],
                'input':item['input'],
                'output':item['output'],
                'data_source':item['data_source']
            })
        

    list_dict_to_json(dataset_deduplicatd,data_out)

def main():
    file_names = os.listdir('./source_data')
    print(file_names)
    for file in file_names:
        if 'cleaned' not in file:
            print('Deduplicating '+file+'...')
            deduplicate('./source_data/'+file,'./source_data/'+file,None)

main()