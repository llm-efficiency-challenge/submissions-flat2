import fasttext
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import argparse

rng = np.random.default_rng(122)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="Qwen/Qwen-14B",
        help="The name or path of the tokenizer.",
    )
    parser.add_argument(
        "--english", action="store_true", help="Use the english part of Open Assistant."
    )
    parser.add_argument(
        "--truncation", action="store_true", help="Truncate the instances of CNN/DM."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push the final dataset on the hub"
    )
    parser.add_argument("--output_path", type=str, default="")
    return parser.parse_args()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, trust_remote_code=True
    )
    # Open Assistant
    oasst1 = load_dataset("OpenAssistant/oasst1")
    i = 0
    len_ = len(oasst1["train"])
    K = []
    while i < len_:
        example = oasst1["train"][i]
        tree_id = example["message_tree_id"]
        role = example["role"]
        j = i + 1
        conversation = []
        dico = {}
        dico[role] = example["text"]
        while j < len_:
            next = oasst1["train"][j]
            if next["message_tree_id"] != tree_id:
                break
            second_role = next["role"]
            dico[second_role] = next["text"]
            if len(dico) == 2:
                conversation.append(dico)
                dico = {}
            j += 1
        i = j
        K.append(conversation)

    K_clean = []
    for i, conv in enumerate(K):
        prompt = conv[0]["prompter"]
        completion = conv[0]["assistant"]
        if "as a language model" in completion.lower():
            continue
        if "sorry" in completion.lower():
            continue
        K_clean.append((prompt, completion))

    if args.english:
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification", filename="model.bin"
        )
        model = fasttext.load_model(model_path)

        count = 0
        W = []
        for prompt, completion in K_clean:
            a, b = model.predict(prompt.strip().split("\n")[0])
            if a[0] == "__label__eng_Latn" and b[0] >= 0.8:
                out = f"Question: {prompt}\nAnswer: {completion}"
                W.append(out)
                print(out)
                print("-" * 200)
                count += 1
        print(count)

        oasst = Dataset.from_dict({"prompt": [e for e in W]})
    else:
        oasst = Dataset.from_dict(
            {
                "prompt": [
                    f"Question: {prompt}\nAnswer: {completion}"
                    for (prompt, completion) in K_clean
                ]
            }
        )

    # CNN/DailyMail
    if args.truncate:
        cnndm = load_dataset("cnn_dailymail", "1.0.0")
        cnndm_sample = cnndm["train"].select(
            rng.choice(len(cnndm["train"]), size=5000, replace=False)
        )

        similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        CNNDM = []
        for i, example in enumerate(cnndm_sample):
            document = example["article"]
            summary = example["highlights"]
            input_ids = tokenizer(document)["input_ids"]
            new_document = tokenizer.decode(input_ids[:512])
            summary = summary.replace(" .", ".")
            out = f"Article: {new_document.strip()}\n\nSummarize the above article in 3 sentences.\n{summary.strip()}"
            doc_embedding = similarity_model.encode(
                new_document, convert_to_tensor=True
            )
            sum_embedding = similarity_model.encode(summary, convert_to_tensor=True)
            sbert_score = util.pytorch_cos_sim(doc_embedding, sum_embedding).item()
            if sbert_score >= 0.8:
                CNNDM.append(out)
    else:
        CNNDM = []
        for i, example in enumerate(cnndm_sample):
            document = example["article"]
            summary = example["highlights"]
            out = f"Article: {document.strip()}\n\nSummarize the above article in 3 sentences.\n{summary.strip()}"
            CNNDM.append(out)

    cnn_dailymail = Dataset.from_dict({"prompt": CNNDM}).shuffle(122)

    # BBQ and TruthfulQA
    BBQ = []
    for config in [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_gender",
        "Race_x_SES",
        "Religion",
        "SES",
        "Sexual_orientation",
    ]:
        bbq = load_dataset("lighteval/bbq_helm", config)
        L = []
        for i, example in enumerate(bbq["train"]):
            context = example["context"]
            question = example["question"]
            choices = example["choices"]
            out = f"Passage: {context}\nQuestion: {question}"
            for i, choice in enumerate(choices):
                out += f"\n{chr(65+i)}. {choice}"
            out += f"\nAnswer: {chr(65+example['gold_index'])}"
            L.append(out)
        L_sample = rng.choice(L, size=int(0.08 * len(L)), replace=False).tolist()
        BBQ.extend(L_sample)

    TQA = []
    truthfulqa = load_dataset("lighteval/truthfulqa_helm")
    for i, example in enumerate(truthfulqa["train"]):
        question = example["question"]
        choices = example["choices"]
        gold_index = example["gold_index"]
        out = f"Question: {question}"
        for i, choice in enumerate(choices):
            out += f"\n{chr(65+i)}. {choice}"
        out += f"\nAnswer: {chr(65+gold_index)}"
        TQA.append(out)

    bbq_tqa = Dataset.from_dict({"prompt": TQA + BBQ}).shuffle(122)

    final_ds = concatenate_datasets([bbq_tqa, oasst, cnn_dailymail]).train_test_split(
        test_size=0.1
    )

    print(final_ds)
    n_tokens = 0
    for example in final_ds["train"]:
        n_tokens += len(tokenizer(example["prompt"])["input_ids"])
    print(f"The total number of steps is {n_tokens//2048//16}.")
    if args.push_to_hub:
        final_ds.push_to_hub(args.output_path)
    else:
        final_ds.save_to_disk(args.output_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
