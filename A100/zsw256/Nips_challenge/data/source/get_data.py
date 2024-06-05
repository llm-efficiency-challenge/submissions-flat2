from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
from flatten_ossat1 import reformat_open_assistant
import random



def list_dict_to_json(input,output_path):
    df = pd.DataFrame(input)
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)
    return

def get_data_from_cnn_dailymail(dataset):
    dataset_formated = []
    print('Reformatting cnn_dailymail...')

    for item in tqdm(dataset):
        dataset_formated.append({
            'instruction':'Summarize the below article in 3 sentences.\n',
            'input':item['article'],
            'output':item['highlights'],
            'data_source':'cnn_dailymail'
        })
    
    return dataset_formated

def get_data_from_lima(dataset):
    dataset_formated = []
    print('Reformatting lima...')

    for item in tqdm(dataset):
        data = item['conversations']
        history=[]
        for i in range(int(len(data[:-2])/2)):
            history.append([data[2*i],data[2*i+1]])
        instruction = data[-2]
        output = data[-1]
        if len(history)!=0:
            dataset_formated.append({
                'instruction':instruction,
                'input':'',
                'output':output,
                'history':history,
                'data_source':'lima/'+item['source']
            })
        else:
            dataset_formated.append({
                'instruction':instruction,
                'input':'',
                'output':output,
                'data_source':'lima/'+item['source']
            })

    return dataset_formated

def get_data_from_platypus(dataset):
    dataset_formated = []
    print('Reformatting open platypus...')
    for item in dataset:
        dataset_formated.append({
            'instruction':item['instruction'],
            'input':item['input'],
            'output':item['output'],
            'data_source':'open_platypus/'+item['data_source']
        })
    return dataset_formated

def get_data_from_gsm8k(dataset,is_socratic=False):
    print('Reformatting gsm8k...')
    dataset_formated=[]
    for item in tqdm(dataset):
        dataset_formated.append({
            'instruction':item['question'],
            'input':'Do it Detailedly!\n' if is_socratic else '',
            'output':item['answer'],
            'data_source':'gsm8k_socratic' if is_socratic else 'gsm8k'
        })
    return dataset_formated

def get_data_from_dolly(dataset):
    print('Reformatting dolly...')
    dataset_formated=[]
    for item in tqdm(dataset):
        dataset_formated.append({
            'instruction':item['instruction'],
            'input':item['context'],
            'output':item['response'],
            'data_source':'dolly/'+item['category']
        })
    return dataset_formated

def get_data_from_codecontests(dataset):
    print('Reformatting code contests...')
    dataset_formated=[]
    for item in tqdm(dataset):
        if len(item['solutions']['solution'])>1:
            dataset_formated.append({
                'instruction':'Solve the next programming problems.\n',
                'input':item['description'],
                'output':item['solutions']['solution'][0],
                'data_source':'codecontests'
            })
    return dataset_formated

def get_data_from_reclor(dataset):
    print('Reformatting reclor...')
    dataset_formated=[]
    for item in tqdm(dataset):
        answer = max(item['answers'], key=len, default='')
        dataset_formated.append({
            'instruction':item['question'],
            'input':item['context'],
            'output':answer,
            'data_source':'reclor'
        })
    return dataset_formated

def get_data_from_tiger(dataset):
    dataset_formated = []
    print('Reformatting tigerbot kaggle...')
    for item in tqdm(dataset):
        dataset_formated.append({
            'instruction':item['instruction'],
            'input':item['input'],
            'output':item['output'],
            'data_source':'tigerbot-kaggle-leetcodesolutions-en-2k'
        })
    return dataset_formated

def get_data_from_AgentInstruct(dataset):
    dataset_formated = []
    print('Reformatting AgentInstruct...')
    subsets = dataset.keys()
    for subset in tqdm(subsets):
        for item in dataset[subset]:
            turns = item['conversations'][:-2]
            instruction = item['conversations'][-2]['value']
            output = item['conversations'][-1]['value']
            history=[]
            for i in range(int(len(turns)/2)):
                history.append([turns[2*i]['value'],turns[2*i+1]['value']])
            dataset_formated.append({
                'instruction':instruction,
                'input':'',
                'output':output,
                'history':history,
                'data_source':'AgentInstruct/'+subset
            })
    return dataset_formated

def get_data_from_competition_math(dataset):
    dataset_formatted=[]
    print('Reformatting competition math...')
    for item in tqdm(dataset):
        dataset_formatted.append({
            'instruction':'Solve the next problem\n',
            'input':item['problem'],
            'output':item['solution'],
            'data_source':'competition_math/'+item['type']
        })
    return dataset_formatted

def get_data_from_sciq(dataset):
    dataset_formatted=[]
    print('Reformatting sciq...')
    for item in tqdm(dataset):
        choices = [item['distractor1'],item['distractor2'],item['distractor3'],item['correct_answer']]
        random.shuffle(choices)
        idx = choices.index(item['correct_answer'])
        instruction = item['question']+'\n[A.'+choices[0]+ ' B.'+choices[1]+ ' C.'+choices[2]+ ' D.'+choices[3]+']'
        dataset_formatted.append({
            'instruction':instruction,
            'input':item['support'],
            'output':chr(idx+ord('A')),
            'data_source':'sciq'
        })
    return dataset_formatted

def get_data_from_openbookqa(dataset):
    dataset_formatted=[]
    print('Reformatting openbookqa...')
    for item in tqdm(dataset):
        choices = item['choices']['text']
        input = item['question_stem']+'\n[A.'+choices[0]+ ' B.'+choices[1]+ ' C.'+choices[2]+ ' D.'+choices[3]+']'
        dataset_formatted.append({
            'instruction':'Answer the following question:\n',
            'input':input,
            'output':item['answerKey'],
            'data_source':'openbookqa'
        })
    return dataset_formatted

def get_data_from_prm800k(dataset):
    dataset_formatted=[]
    print('Reformatting prm800k...')
    for item in tqdm(dataset):
        if len(item['responses'])>1:
            output=''.join(item['responses'])+item['next_response']+'\nAnswer: '+item['answer']
        else:
            output = item['next_response']+'\nAnswer: '+item['answer']
        dataset_formatted.append({
            'instruction':item['instruction'],
            'input':'',
            'output':output,
            'data_source':'prm800k'
        })
    return dataset_formatted

def get_data_from_scienceQA(dataset):
    dataset_formatted=[]
    print('Reformatting scienceQA...')
    for item in tqdm(dataset):
        if item['image'] is None:
            instruction=item['question']+'\nChoices: '+str(item['choices'])
            dataset_formatted.append({
                'instruction':instruction,
                'input':item['lecture'],
                'output':item['solution']+'\nAnswer: '+item['choices'][item['answer']],
                'data_source':'ScienceQA/'+item['subject']
            })
    return dataset_formatted

def get_data_from_truthfulqa(dataset):
    dataset_formatted=[]
    print('Reformatting truthfulqa...')
    for item in tqdm(dataset):
        instruction=item['question']+'\n'+'Choices: ['
        for idx,choice in enumerate(item['choices']):
            instruction+=chr(idx+ord('A'))+'.'+choice+' '
        instruction+=']'
        dataset_formatted.append({
            'instruction':instruction,
            'input':'',
            'output':chr(item['gold_index']+ord('A')),
            'data_source':'truthfulqa'
        })
    return dataset_formatted

def get_data_from_big_bench(dataset):
    dataset_formatted=[]
    print('Reformatting big bench...')
    bigbench_subsets = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']
    for subset in tqdm(bigbench_subsets):
        dataset = load_dataset('lighteval/big_bench_hard',subset)
        for item in dataset['train']:
            dataset_formatted.append({
                'instruction':item['input'],
                'input':'',
                'output':item['target'],
                'data_source':'Big_bench/'+subset
            })
    return dataset_formatted

def main():
    if not os.path.exists('./source_data/'):
        os.mkdir('./source_data/')
    
    dataset = load_dataset("GAIR/lima",token='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')
    dataset = get_data_from_lima(dataset['train'])
    list_dict_to_json(dataset[:-30],'./source_data/lima_cleaned.json')
    list_dict_to_json(dataset[-30:],'./source_data/lima_chat_cleaned.json')
    
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    dataset = get_data_from_cnn_dailymail(dataset['train'])[:10000]
    list_dict_to_json(dataset,'./source_data/cnn_dailymail.json')

    dataset = reformat_open_assistant()
    list_dict_to_json(dataset,'./source_data/open_assistant.json')

    dataset = load_dataset("gsm8k",'main')
    dataset = get_data_from_gsm8k(dataset['train'])
    list_dict_to_json(dataset,'./source_data/gsm8k.json')

    dataset = load_dataset("gsm8k",'socratic')
    dataset = get_data_from_gsm8k(dataset['train'],True)
    list_dict_to_json(dataset,'./source_data/gsm8k_socratic.json')

    dataset = load_dataset("databricks/databricks-dolly-15k")
    dataset = get_data_from_dolly(dataset['train'])
    list_dict_to_json(dataset,'./source_data/dolly.json')

    dataset = load_dataset("deepmind/code_contests")
    dataset = get_data_from_codecontests(dataset['train'])
    list_dict_to_json(dataset,'./source_data/code_contests.json')

    dataset = load_dataset("metaeval/reclor")
    dataset = get_data_from_reclor(dataset['train'])
    list_dict_to_json(dataset,'./source_data/reclor.json')

    dataset = load_dataset("TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k")
    dataset = get_data_from_tiger(dataset['train'])
    list_dict_to_json(dataset,'./source_data/tigerbot_kaggle.json')

    dataset = load_dataset("THUDM/AgentInstruct")
    dataset = get_data_from_AgentInstruct(dataset)
    list_dict_to_json(dataset,'./source_data/agent_instruct_cleaned.json')
    
    dataset = load_dataset("competition_math")
    dataset = get_data_from_competition_math(dataset['train'])
    list_dict_to_json(dataset,'./source_data/competition_math.json')

    dataset = load_dataset("sciq")
    dataset = get_data_from_sciq(dataset['train'])[:4000]
    list_dict_to_json(dataset,'./source_data/sciq.json')

    dataset = load_dataset("openbookqa")
    dataset = get_data_from_openbookqa(dataset['train'])[:2000]
    list_dict_to_json(dataset,'./source_data/openbookqa.json')

    dataset = load_dataset("Birchlabs/openai-prm800k-solutions-only")
    dataset = get_data_from_prm800k(dataset['train'])
    list_dict_to_json(dataset,'./source_data/prm800k.json')

    dataset = load_dataset("derek-thomas/ScienceQA")
    dataset = get_data_from_scienceQA(dataset['train'])
    list_dict_to_json(dataset,'./source_data/scienceqa.json')

    dataset = get_data_from_big_bench(dataset)
    list_dict_to_json(dataset,'./source_data/bigbench.json')

    dataset = load_dataset("lighteval/truthfulqa_helm")
    dataset = get_data_from_truthfulqa(dataset['train'])
    list_dict_to_json(dataset,'./source_data/truthfulqa.json')

    dataset = load_dataset("Rocinante/bbq_cleaned")
    list_dict_to_json(dataset['train'],'./source_data/bbq.json')


main()
