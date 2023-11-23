# PROMPT VARIATION EXAMPLES

# creating prompts vairation for "true/false" questions
python prompt_variation.py

# creating prompts vairation for "yes/no" questions
python prompt_variation.py --yesno -p "../../data/original_prompts.csv" -o "../../data/paraphrased-prompts-modified/all_yesno_short_changeline.json"


# TEST MODELS WITH PROMPTS

# an example for masked models
python model_prompting_script.py -m "cardiffnlp/twitter-roberta-base" -scale "all_truefalse_short_changeline" --accelerate

# examples for causal models

python model_prompting_script.py -m "/shared/2/projects/llm-personas/llama2-bible/checkpoint-5000" -scale "all_truefalse_short_changeline" --causal -BATCH 32
python model_prompting_script.py -m "gpt2" -scale "SIPS_new" --causal -BATCH 32

# an example for LoRA
python model_prompting_script.py -m "/shared/2/projects/llm-personas/llama2-4chan-lora/checkpoint-500" -scale "all_truefalse_short_changeline" --causal -BATCH 32 --lora


# RESULT PROCESSING EXAMPLE

python result_processing.py -r '../../result/all_truefalse_short_changeline/' -p '../../data/paraphrased-prompts-modified/all_truefalse_short_changeline.json' -o "../../result/"
