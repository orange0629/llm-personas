# Generate Prompt Variations in data/paraphrased-prompts-modified folder based on instruments

python prompt_variation.py  #true/false
python prompt_variation.py --yesno -p "../../data/original_prompts.csv" -o "../../data/paraphrased-prompts-modified/all_yesno_short_changeline.json" #yes/no


# Run prompt variations on LLMs

declare -a models=("bigscience/bloomz-560m" "bigscience/bloomz-1b1" "bigscience/bloomz-3b" "bigscience/bloomz-7b1" "gpt2" "togethercomputer/RedPajama-INCITE-7B-Instruct" "tiiuae/falcon-7b-instruct" "meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl")
for i in "${models[@]}"
do
    python run_prompts_on_models.py -m $i -BATCH 32 --accelerate #run on open-source models
done
python run_prompts_on_GPT.py #run on GPT3.5/4

# RESULT PROCESSING

python result_processing.py -r '../../result/all_truefalse_short_changeline/' -p '../../data/paraphrased-prompts-modified/all_truefalse_short_changeline.json' -o "../../result/"
