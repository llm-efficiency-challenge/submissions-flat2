from datasets import load_dataset, Dataset, concatenate_datasets
import random
import re
import pandas

lu = load_dataset("GAIR/lima")['train']
gu = load_dataset("timdettmers/openassistant-guanaco")['train']
mc = load_dataset("euclaise/MiniCoT")['train'].select(range(25000))
mqa = load_dataset("euclaise/mqa", split='train').shuffle(seed=42).select(range(2000))
st = load_dataset("euclaise/symtune_mini", split='train').shuffle(seed=42).select(range(250))
mo = load_dataset("euclaise/mathoverflow-accepted", split='train').filter(lambda x: int(x['score']) > 50).shuffle(seed=42)
sc = load_dataset("euclaise/SciCoT", split='train').shuffle(seed=42).select(range(1000))

def map_lu(row):
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['conversations'][0]},
            {"from": "gpt", "value": row['conversations'][1]}
        ]
    }
lu = lu.map(map_lu, num_proc=8, remove_columns=lu.column_names)

pattern = re.compile(r"### (Human|Assistant):\s+(.*?)(?=(###|$))", re.S)
def map_gu(row):
    matches = re.findall(pattern, row['text'])
    msgs = []
    
    for role, msg, _ in matches:
        msgs.append({
            "from": ("gpt" if role == "Assistant" else "human"),
            "value": msg
        })
    return {'system': "", 'conversations': msgs}
gu = gu.map(map_gu, num_proc=8, remove_columns=gu.column_names)



transitions = [
    "So, the answer is ",
    "Answer: ",
    "Therefore, it's ",
    "Based on the above, the answer is ",
    "Ergo, ",
    "Hence, we conclude ",
    "As a result, ",
    "On this basis, "
    "In conclusion, ",
    "Thus, ",
    "Considering this, we get ",
    "Putting this all together, we get: ",
    "Taking a step back, the result is ",
    "Long story short, ",
    "After all that, it seems that the answer is "
]

def map_mc(row):
    transition = random.choice(transitions)
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['prompt']},
            {"from": "gpt_rationale", "value": f"{row['rationale']}"},
            {"from": "gpt_transition", "value": f"\n{transition}"},
            {"from": "gpt_target", "value": f"{row['target']}"}
        ]
    }
mc = mc.map(map_mc, num_proc=8, remove_columns=mc.column_names)


def map_mqa(row):
    transition = random.choice(transitions)
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['msg']},
            {"from": "gpt_rationale", "value": ""},
            {"from": "gpt_transition", "value": transition},
            {"from": "gpt_target", "value": row['resp_correct']}
        ]
    }
mqa = mqa.map(map_mqa, num_proc=8, remove_columns=mqa.column_names)

def map_st(row):
    transition = random.choice(transitions)
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['question']},
            {"from": "gpt_rationale", "value": ""},
            {"from": "gpt_transition", "value": transition},
            {"from": "gpt_target", "value": row['answer']}
        ]
    }
st = st.map(map_st, num_proc=8, remove_columns=st.column_names)


def map_mo(row):
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['parent_title'] + '\n\n' + row['parent_body']},
            {"from": "gpt", "value": row['body']}
        ]
    }
mo = mo.map(map_mo, num_proc=8, remove_columns=mo.column_names)

def map_sc(row):
    transition = random.choice(transitions)
    return {
        'system': "",
        'conversations': [
            {"from": "human", "value": row['prompt']},
            {"from": "gpt_rationale", "value": f"{row['rationale']}"},
            {"from": "gpt_transition", "value": f"\n{transition}"},
            {"from": "gpt_target", "value": f"{row['target']}"}
        ]
    }
sc = sc.map(map_sc, num_proc=8, remove_columns=sc.column_names)


ds = concatenate_datasets([lu, gu, mc, mqa, st, mo, sc]).shuffle(seed=42)


print(ds)
ds.save_to_disk("ds")
