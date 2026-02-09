
##  Usage

### Setup

```bash
conda create -n TabooRAG python=3.11
```
```bash
conda activate TabooRAG
```
```bash
pip install beir openai
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install --upgrade charset-normalizer
pip install "fschat[model_worker,webui]"
```

### Prepare Dataset
When running our code, the datasets will be automatically downloaded and saved in `datasets`. You could also run this line to manually download datasets.

```bash
python prepare_dataset.py
```

### Set API key

Our experiments use GPT-5.2, GPT-5-mini, DeepSeek-V3.2, Qwen3-32B, Gemma-3-12B-IT, Ministral-3-8B-Instruct-2512, and Meta-Llama-3-8B-Instruct. You can configure the API or local settings in the `model_configs` file.

GPT example:
```json
{
    "model_info":{
        "provider":"gpt",
        "name":"gpt-5.2"
    },
    "api_key_info":{
        "api_keys":[
            "Your api key here"
        ],
        "api_key_use": 0
    }
}
```

vLLM example (You can set the port number in the `src/models/vLLM_local.py` file.): 
```json
{
    "model_info":{
        "provider":"vllm-local",
        "name":"Qwen-3-32B"
    },
}
```


### Get Clean Response
```
python get_clean_response.py --llm_model gpt-5.2
```


### Generate Blocking Document
Execute `gen_attack_docs.py` to generate Blocking document.

Example:
```
python gen_attack_docs.py --dataset nq
```




### Run
Set hyperparameters in `run.py`:

Execute **run.py** to reproduce experiments.

```bash
python run.py
```



