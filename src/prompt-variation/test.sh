# TEST MODELS WITH PROMPTS

# an example for masked models
python model_prompting_script.py -m "cardiffnlp/twitter-roberta-base" -scale "ACI" --accelerate

# examples for causal models

python model_prompting_script.py -m "/shared/2/projects/llm-personas/llama2-bible/checkpoint-5000" -scale "SIPS_new" --causal -BATCH 32
python model_prompting_script.py -m "gpt2" -scale "SIPS_new" --causal -BATCH 32

# an example for LoRA
python model_prompting_script.py -m "/shared/2/projects/llm-personas/llama2-4chan-lora/checkpoint-500" -scale "EPQ_new" --causal -BATCH 32 --lora
