import pandas as pd
from transformers import AutoTokenizer

# 假设的tokenizer，实际使用时需要根据具体模型加载
tokenizer = AutoTokenizer.from_pretrained('your-model-path')

def tokenize_dialog(dialog, tokenizer):
    """根据对话内容生成输入和标签，处理特殊标记"""
    dialog_tokens = [tokenizer.encode(f"[INST] {entry['content']} [/INST]" if entry['role'] == 'user' else f"{entry['content']}", add_special_tokens=False) for entry in dialog]
    labels_tokens = [-100 if entry['role'] == 'user' else tokenizer.encode(entry['content'], add_special_tokens=False) for entry in dialog]

    # Flatten the list of tokens
    input_ids = [token for sublist in dialog_tokens for token in sublist]
    labels = [label for sublist in labels_tokens for label in sublist]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

def process_excel_file(file_path):
    """從Excel取資料"""
    data = pd.read_excel(file_path)
    dialogs = []

    for index, row in data.iterrows():
        dialog = [
            {"role": "user", "content": row["Q"]},
            {"role": "assistant", "content": row["A"]}
        ]
        dialogs.append(dialog)

    return dialogs

def main(file_path, tokenizer):
    """執行數據加載、轉化與tokenize過程"""
    dialogs = process_excel_file(file_path)
    tokenized_data = []

    for dialog in dialogs:
        tokenized_result = tokenize_dialog(dialog, tokenizer)
        tokenized_data.append(tokenized_result)


    print(tokenized_data)


if __name__ == "__main__":
    main('/path/to/your/excel.xlsx', tokenizer)
