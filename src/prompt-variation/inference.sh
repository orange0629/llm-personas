#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "bigscience/bloomz-560m" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "bigscience/bloomz-1b1" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "bigscience/bloomz-3b" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "gpt2" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "togethercomputer/RedPajama-INCITE-7B-Instruct" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "tiiuae/falcon-7b-instruct" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "mistralai/Mistral-7B-Instruct-v0.1" -scale "instruments_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "/shared/4/models/llama2/pytorch-versions/llama-2-7b" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "/shared/4/models/llama2/pytorch-versions/llama-2-13b" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "/shared/4/models/llama2/pytorch-versions/llama-2-7b-chat" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "/shared/4/models/llama2/pytorch-versions/llama-2-13b-chat" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,2,3 python model_prompting_script_version2.py -m "bigscience/bloomz-7b1" -scale "all_truefalse_short_changeline" -BATCH 32 --accelerate
#CUDA_VISIBLE_DEVICES=1,3,4 python model_prompting_script_version2.py -m "mosaicml/mpt-7b-instruct" -scale "instruments_truefalse_short" -BATCH 32 --accelerate

#CUDA_VISIBLE_DEVICES=1 python model_prompting_script_version3.py -m "google/flan-t5-small" -BATCH 32
#CUDA_VISIBLE_DEVICES=1 python model_prompting_script_version3.py -m "google/flan-t5-base" -BATCH 32
#CUDA_VISIBLE_DEVICES=1 python model_prompting_script_version3.py -m "google/flan-t5-large" -BATCH 32
#CUDA_VISIBLE_DEVICES=1,2 python model_prompting_script_version3.py -m "google/flan-t5-xl" -BATCH 32 --accelerate

CUDA_VISIBLE_DEVICES=2,3,4,5 python model_prompting_script_version3.py -m "google/flan-t5-xxl" -BATCH 32 --accelerate