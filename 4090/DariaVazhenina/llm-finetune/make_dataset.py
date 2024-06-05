import os
import pandas as pd
import random
import re

from datasets import load_dataset
from tqdm import tqdm

output_dir = "./data"
file_name = "./combined_dataset.csv"

n_imdb = 4000
n_cnndm = 4000
n_dolly = 15000
n_mmlu = 4000
n_msmarco = 500
n_civil = 4000
n_babi = 4000
n_narrative = 4000
n_xsum = 4000
n_raft = 500


def preprocess(texts):
    pattern_remained_chars = r'[^A-Za-z0-9.,!?:;\'"“”\-\[\]{}()…_$%&+\s]'
    pattern_symbol_seq     = r"([,!?:;'\-“”\[\]{}()…_])\1{2,}"
    texts = [re.sub('<.*?>', ' ', t) for t in texts]               # Delete html tags
    texts = [re.sub('http\S+', ' ', t) for t in texts]             # Delete links
    texts = [re.sub(pattern_remained_chars, '', t) for t in texts] # Leave only alphanumeric characters, common symbols, any space character
    texts = [re.sub(pattern_symbol_seq, r"\1", t) for t in texts]  # Replace three or more repetitions of the same symbol with a single symbol
    texts = [re.sub('\s+', ' ', t) for t in texts]                 # Replace a sequence of whitespace with a single whitespace character
    texts = [t.strip() for t in texts]                              # strip

    return texts

# df for output
df = pd.DataFrame(columns=["prompt", "prompt_with_label", "dataset", "task", "context", "instruction", \
                           "Choice_A", "Choice_B", "Choice_C", "Choice_D", "label"])

# ----- IMDB --------------------
imdb = load_dataset("imdb")["train"]
context_tmp = imdb["text"]
label_tmp = imdb["label"]

# Shuffle
index_list = list(range(len(context_tmp)))
random.shuffle(index_list)
context = [context_tmp[i] for i in index_list]
label = [label_tmp[i] for i in index_list]

# slice
context = context[:n_imdb]
label = label[:n_imdb]

# Preprocess
context = preprocess(context)
label = ["Negative" if l==0 else "Positive" for l in label]
            
for c, l in tqdm(zip(context, label), total=len(label)):
    prompt = f"{c} Sentiment: "
    prompt_with_label =  f"{c} Sentiment: {l}"
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'imdb',
        'task':'sentiment classification',
        'context': c,
        'label': l
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- CNNDM --------------------
cnndm = load_dataset("cnn_dailymail", "3.0.0")["train"]
context = cnndm["article"][:n_cnndm]
context = preprocess(context)
label = cnndm["highlights"][:n_cnndm]
label = preprocess(label)

for c, l in tqdm(zip(context, label), total=len(label)):
    prompt = f"Summarize the given document. Document: {c} Summary: "
    prompt_with_label =  f"Summarize the given document. Document: {c} Summary: {l}"
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'cnndm',
        'task':'summarization',
        'context': c,
        'label': l
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- Dolly --------------------    
dolly = load_dataset("databricks/databricks-dolly-15k")["train"]
context = dolly["context"][:n_dolly]
context = preprocess(context)
instruction = dolly["instruction"][:n_dolly]
label = dolly["response"][:n_dolly]
label = preprocess(label)
task = dolly["category"][:n_dolly]

for idx, (c, i, l, t) in tqdm(enumerate(zip(context, instruction, label, task)), total=len(label)):
    if t == "brainstorming":
        continue  # No related task in HELM
    elif t == "classification":
        # The classification task in HELM is a RAFT dataset, but the nature of the task is very different 
        # because it is a classification of whether the text is ADE related or not
        continue  # No related task in HELM
    elif t == "closed_qa":
        prompt = "Context: " + c + " " + "Question: " + i + " " + "Answer: "
        prompt_with_label = "Context: " + c + " " + "Question: " + i + " " + "Answer: " + l 
    elif t == "creative_writing": 
        continue  # No related task in HELM
    elif t == "general_qa":
        # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
        prompt = "Question: " + i + " " + "Answer: "
        prompt_with_label = "Question: " + i + " " + "Answer: " + l 
    elif t == "information_extraction":
        # Since MS MARCO is a two-class classification task, I decided on the prompt as a QA task with context.
        # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
        prompt = "Context: " + c + " " + "Question: " + i + " " + "Answer: "
        prompt_with_label = "Context: " + c + " " + "Question: " + i + " " + "Answer: " + l 
    elif t == "open_qa":
        # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/natural_qa_scenario.py
        prompt = "Question: " + i + " " + "Answer: "
        prompt_with_label = "Question: " + i + " " + "Answer: " + l 
    elif t == "summarization":
        # https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/summarization_scenario.py
        prompt = "Summarize the given document." + " " + "Document: " + c + " " + "Summary: " 
        prompt_with_label = "Summarize the given document." + " " + "Document: " + c + " " + "Summary: " + l 
    else:
        continue

    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'dolly',
        'task': t,
        'context': c,
        'instruction': i,
        'label': l,
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- MMLU --------------------
mmlu = load_dataset("cais/mmlu", "all")["auxiliary_train"]
instruction = mmlu["question"][:n_mmlu]
choices = mmlu["choices"][:n_mmlu]
label = mmlu["answer"][:n_mmlu]

instruction = preprocess(instruction)

mapping_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

for i, c, l in tqdm(zip(instruction, choices, label), total=len(label)):
    opt_a = c[0]
    opt_b = c[1]
    opt_c = c[2]
    opt_d = c[3]

    l = mapping_dict.get(l)
    prompt = f"{i} A. {opt_a} B. {opt_b} C. {opt_c} D. {opt_d} Answer: "
    prompt_with_label = f"{i} A. {opt_a} B. {opt_b} C. {opt_c} D. {opt_d} Answer: {l}"
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'MMLU',
        'task':'QA',
        'instruction': i,
        'Choice_A': opt_a,
        'Choice_B': opt_b,
        'Choice_C': opt_c,
        'Choice_D': opt_d,
        'label': l
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- MS-MARCO --------------------
msmarco = load_dataset("ms_marco", "v1.1")["train"]
passage = msmarco["passages"][:n_msmarco]
query = msmarco["query"][:n_msmarco]
answer = msmarco["answers"][:n_msmarco]

for p, q, a in tqdm(zip(passage, query, answer), total=len(passage)):
    context = p["passage_text"]
    is_selected = p["is_selected"]
    for c, l in zip(context, is_selected):
        if l == 1:
            label = "Yes"
        else:
            label = "No"
        
        c = preprocess([c])[0]
        prompt = f"Passage: {c} Query: {q} Does the passage answer the query? Answer: "
        prompt_with_label = f"Passage: {c} Query: {q} Does the passage answer the query? Answer: {label}"
    
        new_row = {
            'prompt': prompt,
            'prompt_with_label': prompt_with_label,
            'dataset':'msmarco',
            'task':'Information retrieval',
            'context': c,
            'instruction':q ,
            'label': label
        }

        df_new_row = pd.DataFrame([new_row])
        df = pd.concat([df, df_new_row], ignore_index=True)

# ----- CivilComments --------------------
civil = load_dataset("lighteval/civil_comments_helm", "all")["train"]
context = civil["text"][:n_civil]
context = preprocess(context)
label = civil["gold"][:n_civil]
label = preprocess(label)

for c, l in tqdm(zip(context, label), total=len(label)):
    prompt = f"{c} Question: Is the passage above toxic? "
    prompt_with_label =  f"{c} Question: Is the passage above toxic? {l}"
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'civil_comments',
        'task':'Toxic classification',
        'context': c,
        'label': l
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- Babi QA --------------------
babi = load_dataset("facebook/babi_qa", "en-10k-qa1")["train"]
story = babi["story"][:n_babi]

for s in tqdm(story, total=len(story)):
    text = s["text"]
    answer = s["answer"]
    context = ""
    for i, (t, a) in enumerate(zip(text, answer)):
        context += t + " "
        if i+1 == len(answer):
            break
        context += a + " "
    context = preprocess([context])[0]
    
    prompt = context
    prompt_with_label = context + " " + a
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'babi_qa',
        'task':'reasoning',
        'context': context,
        'label': a
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- Narrative QA --------------------
narrative_qa = load_dataset("narrativeqa")["train"]

document = narrative_qa["document"][:n_narrative]
question = narrative_qa["question"][:n_narrative]
answers = narrative_qa["answers"][:n_narrative]

for d, q, a in tqdm(zip(document, question, answers), total=len(document)):
    context = d["summary"]["text"]
    question = q["text"]
    answer = a[0]["text"]
    
    prompt = f"{context} Question: {question} Answer: " 
    prompt_with_label = f"{context} Question: {question} Answer: {answer}" 
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'narrative_qa',
        'task':'narrative_qa',
        'context': context,
        'instruction': question,
        'label': answer
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- XSUM --------------------
xsum = load_dataset("EdinburghNLP/xsum")["train"]
context = xsum["document"][:n_xsum]
context = preprocess(context)
label = xsum["summary"][:n_xsum]
label = preprocess(label)

for c, l in tqdm(zip(context, label), total=len(context)):
    prompt = f"Summarize the given document. Document: {c} Summary: "
    prompt_with_label =  f"Summarize the given document. Document: {c} Summary: {l}"
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'xsum',
        'task':'summarization',
        'context': c,
        'label': l
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)

# ----- RAFT --------------------
raft_ade = load_dataset("ought/raft", "ade_corpus_v2")["train"]
context_ade = raft_ade["Sentence"]
label_ade = raft_ade["Label"]
label_ade = ["ADE-related" if l==1 else "not ADE-related" for l in label_ade]

raft_neurips = load_dataset("ought/raft", "neurips_impact_statement_risks")["train"]
context_neurips = raft_neurips["Impact statement"]
label_neurips = raft_neurips["Label"]
label_neurips = ["doesn't mention a harmful application" if l==1 else "mentions a harmful application" for l in label_neurips]

raft_hate = load_dataset("ought/raft", "tweet_eval_hate")["train"]
context_hate = raft_hate["Tweet"]
label_hate = raft_hate["Label"]
label_hate = ["hate speech" if l==1 else "not hate speech" for l in label_hate]

# slice
context_all = context_ade + context_neurips + context_hate
label_all = label_ade + label_neurips + label_hate
context_all = context_all[:n_raft]
label_all = label_all[:n_raft]

# Preprocess
context_all = preprocess(context_all)
            
for c, l in tqdm(zip(context_all, label_all), total=len(label_all)):
    prompt = f"Sentence: {c} Label: "
    prompt_with_label =  f"Sentence: {c} Label: {l}"
    
    new_row = {
        'prompt': prompt,
        'prompt_with_label': prompt_with_label,
        'dataset':'raft',
        'task':'classification',
        'context': c,
        'label': l
    }
    df_new_row = pd.DataFrame([new_row])
    df = pd.concat([df, df_new_row], ignore_index=True)
    
df["prompt"] = df["prompt"].str.strip()
df["prompt_with_label"] = df["prompt_with_label"].str.strip()
df.to_csv(os.path.join(output_dir, file_name), index=False)
