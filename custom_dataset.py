import copy
import datasets
import itertools
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from zipfile import ZipFile

import os

B_INST, E_INST = "[INST]", "[/INST]"
file_path = '/dataset/little_flower/data/train.xlsx'


def load_and_prepare_excel(shit):
    # 傳入的shit用不到
    print("file is")
    print(file_path)
    print(os.path.isfile(file_path))
    data = pd.read_excel(file_path, engine='openpyxl')
    threads = []
    for index, row in data.iterrows():
        thread = [{'content': row['Q'], 'role': 'user'}, {'content': row['A'], 'role': 'assistant'}]
        threads.append(thread)
    return threads


def tokenize_dialog(dialog, tokenizer):
    if tokenizer.vocab_size >= 128000:
        dialog_tokens = tokenizer.apply_chat_template(dialog)
        dialog_tokens = dialog_tokens[:-4]
        eot_indices = [i for i, n in enumerate(dialog_tokens) if n == 128009]
        labels = copy.copy(dialog_tokens)
        last_idx = 0
        for n, idx in enumerate(eot_indices):
            if n % 2 == 1:
                last_idx = idx
            else:
                labels[last_idx:idx + 1] = [-100] * (idx - last_idx + 1)

        dialog_tokens = [dialog_tokens]
        labels_tokens = [labels]
    else:
        prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {str(prompt['content']).strip()} {E_INST}",
                                          add_special_tokens=False) for prompt in dialog[::2]]
        answer_tokens = [
            tokenizer.encode(f"{str(answer['content']).strip()} {tokenizer.eos_token}", add_special_tokens=False) for
            answer in dialog[1::2]]
        dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
        labels_tokens = [len(c) * [-100, ] if i % 2 == 0 else c for i, c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):
    threads = load_and_prepare_excel(dataset_config.file)
    all_dialogs = []
    for thread in threads:
        dialog_data = tokenize_dialog(thread, tokenizer)
        all_dialogs.append(dialog_data)

    dataset = datasets.Dataset.from_dict({
        "input_ids": [d['input_ids'] for d in all_dialogs],
        "labels": [d['labels'] for d in all_dialogs],
        "attention_mask": [d['attention_mask'] for d in all_dialogs]
    })

    return dataset