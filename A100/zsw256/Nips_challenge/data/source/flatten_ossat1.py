# This script is copied from https://github.com/h2oai/h2ogpt/blob/45e6183171fb16691ad7d3ab006fad973f971e98/create_data.py

import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import _num_samples
import numpy as np
import json
from datasets import load_dataset
from tqdm import tqdm

def parallel_apply(df, func, n_jobs=-1, **kwargs):
    """ Pandas apply in parallel using joblib.
    Uses sklearn.utils to partition input evenly.

    Args:
        df: Pandas DataFrame, Series, or any other object that supports slicing and apply.
        func: Callable to apply
        n_jobs: Desired number of workers. Default value -1 means use all available cores.
        **kwargs: Any additional parameters will be supplied to the apply function

    Returns:
        Same as for normal Pandas DataFrame.apply()

    """

    if effective_n_jobs(n_jobs) == 1:
        return df.apply(func, **kwargs)
    else:
        ret = Parallel(n_jobs=n_jobs)(
            delayed(type(df).apply)(df[s], func, **kwargs)
            for s in gen_even_slices(_num_samples(df), effective_n_jobs(n_jobs)))
        return pd.concat(ret)

def test_add_open_assistant(save_json=True):
    """
    Flatten tree structure into one row per path from root to leaf
    Also turn into human_bot prompting format:
        <human>: question <bot>: answer <human>: question2 <bot>: answer2 Etc.
    Also saves a .json locally as side-effect
    returns list of dicts, containing intput, prompt_type and source
    """
    data_file = "OpenAssistant/oasst1"
    ds = load_dataset(data_file)
    df = pd.concat([ds['train'].to_pandas()], axis=0)
    rows = {}
    message_ids = df['message_id'].values.tolist()
    message_tree_ids = df['message_tree_id'].values.tolist()
    parent_ids = df['parent_id'].values.tolist()
    texts = df['text'].values.tolist()
    roles = df['role'].values.tolist()
    langs = df['lang'].values.tolist()
    print("Collecting All Trees From OpenAssistant/oasst1...")
    for i in tqdm(range(df.shape[0])):
        message_id = message_ids[i]
        message_tree_id = message_tree_ids[i]
        parent_id = parent_ids[i]
        text = texts[i]
        role = roles[i]
        new_data = ('<human>: ' if role == 'prompter' else '<assistant>: ') + text
        entry = dict(message_id=message_id, parent_id=parent_id, text=new_data, lang=langs[i])
        if message_tree_id not in rows:
            rows[message_tree_id] = [entry]
        else:
            rows[message_tree_id].append(entry)
    all_rows = []

    print("Building Samples...")
    for node_id in tqdm(rows):
        # order responses in tree, based on message/parent relationship
        conversations = []
        list_msgs = rows[node_id]
        # find start
        while len(list_msgs):
            for i, leaf in enumerate(list_msgs):
                found = False
                parent_id = leaf['parent_id']
                if parent_id is None:
                    # conversation starter
                    conversations.append(leaf)
                    found = True
                else:
                    for conv in conversations:
                        # find all conversations to add my message to
                        if parent_id in conv['message_id'] and parent_id != conv['message_id'][-len(parent_id):]:
                            # my message doesn't follow conversation
                            continue
                        if parent_id == conv['message_id'][-len(parent_id):]:
                            # my message follows conversation, but fork first, so another follow-on message can do same
                            conversations.append(conv.copy())
                            conv['text'] += f"""
{leaf['text']}
"""
                            conv['message_id'] += leaf['message_id']
                            found = True
                            break
                if found:
                    # my content was used, so nuke from list
                    del list_msgs[i]
                    break

        # now reduce down to final conversations, find the longest chains of message ids
        for i, conv in enumerate(conversations):
            for j, conv2 in enumerate(conversations):
                if i == j:
                    continue
                if conv['message_id'] and conv2['message_id']:
                    assert conv['message_id'] != conv2['message_id']
                    # delete the shorter conversation, if one contains the other
                    if conv['message_id'] in conv2['message_id']:
                        conv['message_id'] = None
                    if conv2['message_id'] in conv['message_id']:
                        conv2['message_id'] = None
        conversations = [c for c in conversations if c['message_id']]
        all_rows.extend([dict(input=c['text'], prompt_type='plain', source=data_file, lang=c['lang']) for c in conversations])
    print(len(all_rows))
    if save_json:
        with open(data_file.lower().replace("/", "_") + ".json", "w") as f:
            f.write(json.dumps(all_rows, indent=2))
    return all_rows

def reformat_open_assistant():
    open_assistant = test_add_open_assistant(save_json=False)
    formatted_dataset = []
    print('Reformatting open assistant...')
    for item in tqdm(open_assistant):
        if item['lang']=='en':
            turns = item['input'].split('<human>: ')
            if '<assistant>' not in turns[-1]:
                turns = turns[:-1]
                if '<assistant>' not in turns[-1]:
                    continue
            instruction=turns[-1].split('<assistant>: ')[0]
            output=turns[-1].split('<assistant>: ')[-1]
            history=[]
            for turn in turns[:-1]:
                if len(turn.split('<assistant>: '))>1:
                    history.append(turn.split('<assistant>: '))
            formatted_dataset.append({
                'instruction':instruction,
                'input':'',
                'output':output,
                'history':history,
                'data_source':'ossat1'
            })
    return formatted_dataset