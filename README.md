# little_flower_training #


### 1. 先確認資料夾有裝git ###
```
git init
git rev-parse --git-dir
```
有回傳 .git 表示成功


### 2. 取得模型 ###
本次使用的模型由 huggingface 下載：[TAIDE-LX-7B](https://huggingface.co/taide/TAIDE-LX-7B-Chat  "TAIDE-LX-7B")
```
mkdir models && cd models
git lfs install
git clone https://huggingface.co/taide/TAIDE-LX-7B-Chat
```

### 3. 取得 llama-recipes ###
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes

```


### 4. 下載 llama-recipes 必要套件 ###
```
git clone git@github.com:meta-llama/llama-recipes.git
cd llama-recipes
```
```
pip install -U pip setuptools
pip install -e .
pip install requirent
```

### 5. 測試一下模型 ###
舉例：
```
import torch
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM, BitsAndBytesConfig

model_id = "/dataset/little_flower/model/TAIDE-LX-7B-Chat"
tokenizer = LlamaTokenizer.from_pretrained(model_id)


# 設定batch size
batch_size = 2

# 加載模型
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

question = "今天天氣真好~"
system_prompt = "<s>[INST] " + question + " [/INST]"

# 使用tokenizer轉換prompt並送到模型
model_input = tokenizer(system_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

```
### 6. 準備微調資料 ###
參考說明：[recipes 中 finetuning 文檔](https://github.com/meta-llama/llama-recipes/blob/main/recipes/finetuning/datasets/README.md)
+ 到 llama-recipes/recipes/finetuning/datasets/custom_dataset.py 修改def get_custom_dataset(dataset_config, tokenizer, split: str) 的內容

+ 到 little_flower/llama-recipes/src/llama_recipes/configs/datasets.py 修改 custom_dataset 的 configs


### 7. 進行微調 ###

```
python -m finetuning.py  --use_peft --peft_method lora --quantization --use_fp16 --model_name /content/little_flower/model/TAIDE-LX-7B-Chat --output_dir  /content/little_flower/model_train
```
