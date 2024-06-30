import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map  # somewhat experimental

project_path_base = "/home/leczhang/research/llm-personas"

CAUSAL = True

#SCALE_LIST = ['OCEAN', 'MFT', 'RVS', 'SIBS', 'HCS', 'UAS', 'RCBS', 'ACI', 'EPQ', 'BSCTM',
#'BSSS', 'SBI', 'PDBS', 'BCQ', 'CCS', 'AIS', 'BRS', 'MHBS', 'ONBGS', 'IS', 'ERUS', 'PSNS',
#'MAS', 'ATPLS', 'FIS', 'MAQ', 'NBI', 'SSIS', 'MMMS', 'VES', 'DAI', 'UWS', 'PES', 'RIQ', 'TLS']

SCALE_LIST = ['prompt_sensitivity/Truefalse_short-statement_double-bar-separated_colon-zero-space-ending_answer-asking', 
              'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-double-space-ending_answer-asking',
              'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-linebreak-ending_answer-asking',
              'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-linebreak-ending_response-asking',
              'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-single-space-ending_answer-asking',
              'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-zero-space-ending_answer-asking',
              'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_question-mark-linebreak-ending_answer-asking',
              'prompt_sensitivity/Truefalse_short-statement_single-space-separated_colon-zero-space-ending_answer-asking',
              'prompt_sensitivity/Truefalse_short-statement_triple-sharp-linebreak-separated_colon-linebreak-ending_answer-asking',
              'all_truefalse_short_changeline',
              'all_yesno_short_changeline'
              ]

SCALE_LIST = ['prompt_with_personas/abuse_all_truefalse_short_changeline',
              'prompt_with_personas/classism_all_truefalse_short_changeline',
              'prompt_with_personas/conspiracy_all_truefalse_short_changeline',
              'prompt_with_personas/control_all_truefalse_short_changeline',
              'prompt_with_personas/propolice_all_truefalse_short_changeline',
              'prompt_with_personas/protestant_all_truefalse_short_changeline',
              'prompt_with_personas/reciprocal_all_truefalse_short_changeline',
              'prompt_with_personas/societal_all_truefalse_short_changeline',
              'prompt_with_personas/spiritual_all_truefalse_short_changeline',
              'prompt_with_personas/ssi_all_truefalse_short_changeline'
              ]

SCALE_LIST = ['prompt_with_personas_v3/extrovert-v1_all_truefalse_short_changeline',
              'prompt_with_personas_v3/extrovert-v2_all_truefalse_short_changeline',
              'prompt_with_personas_v3/normal_all_truefalse_short_changeline',
              ]

SCALE_LIST = ['prompt_with_personas_v4/agreeable_all_truefalse_short_changeline',
              'prompt_with_personas_v4/conscientious_all_truefalse_short_changeline',
              ]

SCALE_LIST = ['prompt_with_personas_v5/convervatism_all_truefalse_short_changeline',
              'prompt_with_personas_v5/imagination_all_truefalse_short_changeline',
              'prompt_with_personas_v5/neuroticism_all_truefalse_short_changeline',
              ]

SCALE_LIST = ['uber_all_truefalse_short_changeline',
              ]

SCALE_LIST = [#'prompt_sensitivity/Truefalse_short-statement_double-bar-separated_colon-zero-space-ending_answer-asking', 
              #'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-double-space-ending_answer-asking',
              #'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-linebreak-ending_answer-asking',
              #'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-linebreak-ending_response-asking',
              #'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-single-space-ending_answer-asking',
              #'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_colon-zero-space-ending_answer-asking',
              #'prompt_sensitivity/Truefalse_short-statement_linebreak-separated_question-mark-linebreak-ending_answer-asking',
              #'prompt_sensitivity/Truefalse_short-statement_single-space-separated_colon-zero-space-ending_answer-asking',
              #'prompt_sensitivity/Truefalse_short-statement_triple-sharp-linebreak-separated_colon-linebreak-ending_answer-asking',
              #'all_truefalse_short_changeline',
              'prompt_with_personas_v3/extrovert-v1_all_truefalse_short_changeline',
              #'prompt_with_personas_v3/extrovert-v2_all_truefalse_short_changeline',
              'prompt_with_personas_v3/normal_all_truefalse_short_changeline',
              'prompt_with_personas_v4/agreeable_all_truefalse_short_changeline',
              'prompt_with_personas_v4/conscientious_all_truefalse_short_changeline',
              'prompt_with_personas_v5/convervatism_all_truefalse_short_changeline',
              'prompt_with_personas_v5/imagination_all_truefalse_short_changeline',
              'prompt_with_personas_v5/neuroticism_all_truefalse_short_changeline',
              'uber_all_truefalse_short_changeline',
              ]

seq2seq = ["t5-small", "t5-base", "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"]


def load_prompts(prompt_path):
    truth_prompt_text_list = []
    judgment_prompt_text_list = []
    p2class = {}
    with open(prompt_path) as f:
        prompt_dict = pd.read_json(f)

        for prompt_row in prompt_dict.itertuples():
            text = prompt_row.text

            p2class[text] = prompt_row.opt_class

            if p2class[text] == "truth":
                truth_prompt_text_list.append(text)
            else:
                judgment_prompt_text_list.append(text)

    return truth_prompt_text_list, judgment_prompt_text_list


def flatten(l):
    return [item for sublist in l for item in sublist]


def evaluate_with_prompts_causal(batched_instances, word_ids, model, tokenizer, SEQ2SEQ=False):
    probs = {}
    if SEQ2SEQ:
        # a vocab dictionary word:id
        target_dict = {tokenizer.decode(i[0]): i[0] for i in word_ids}
    else:
        target_dict = {tokenizer.decode(i): i for i in word_ids if ((type(i) == int) or (len(i) == 1))}  # ONLY next token
    count = 0

    num_beams = False
    if LONGER_SEQS:
        num_beams = NUM_BEAMS
    num_return_sequences = NUM_SEQS

    # generate the next tokens based on prompts
    for b in tqdm(batched_instances):
        with torch.no_grad():

            encoded = tokenizer(b, max_length=512, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            input_data = {
                "input_ids": encoded.input_ids,
                "attention_mask": encoded.attention_mask,
            }

            # Generate output
            generated = model.generate(
                **input_data,
                pad_token_id=tokenizer.pad_token_id,
                # no_repeat_ngram_size=1,
                return_dict_in_generate=True,
                # num_beams=num_beams,
                output_scores=True,
                remove_invalid_values=True,
                # num_return_sequences=num_return_sequences,
                renormalize_logits=True,
                max_length=MAX_LENGTH
            )

        # calculate the probability
        for instance in range(len(b)):  # iterate through instances
            prompt = b[instance]
            next_token_logits = generated.scores[0][instance]

            ## add softmax here (By Bangzhao)
            next_token_logits = softmax(next_token_logits)
            if LONGER_SEQS:
                top_probs = {tokenizer.decode(seq, skip_special_tokens=True): score
                             for seq,score in zip(generated['sequences'], generated['sequences_scores'])}
            else:
                ids = next_token_logits.topk(k=50).indices
                top_probs = {}
                for id in ids:
                    if(tokenizer.decode(id)) in top_probs:
                        top_probs[tokenizer.decode(id)] += float(next_token_logits[id].cpu().numpy())
                    else:
                        top_probs[tokenizer.decode(id)] = float(next_token_logits[id].cpu().numpy())
                # get the probability of target words if they are not top 50
                #for tok, id in target_dict.items():
                #    # get the probability of the next token id
                #    prob = next_token_logits[id]
                #    if "\n" in tok:
                #        top_probs[tok.replace("\n", "\\n")] = float(prob.cpu().numpy())
                #    else:
                #        top_probs[tok] = float(prob.cpu().numpy())

            probs[prompt] = top_probs
        count += 1

    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-scale", help="scale name", default=None)
    parser.add_argument("-d", help="device", type=int, default=0)
    parser.add_argument("-BATCH", help="device", type=int, default=1)
    parser.add_argument("-model", help="model to evaluate", type=str, default=None)
    parser.add_argument("--accelerate", help="use accelerate", action='store_true')
    parser.add_argument("--record_longer_seqs", help="store sequence scores (instead of just next token probabilities)", action='store_true')
    parser.add_argument("-num_beams", help="number of beams for beam search decoding", type=int, default=10)
    parser.add_argument("--lora", help="whether load lora weights", action='store_true')
    parser.add_argument("-num_returned_seqs", help="number of returned sequences for beam search decoding", type=int, default=1)

    args = parser.parse_args()
    assert (args.num_beams >= args.num_returned_seqs)

    if args.scale is None:
        print("No scale given!")
    # what I add (By Bangzhao)
    else:
        SCALE_LIST = [args.scale]

    ACCELERATE = args.accelerate
    LORA = args.lora
    LONGER_SEQS = args.record_longer_seqs
    print(args.model)

    if args.model is None:
        print("No model selected!")
    model = args.model

    NUM_BEAMS = args.num_beams
    NUM_SEQS = args.num_returned_seqs

    formatted_results_path_base = project_path_base + "/result/"
    templates_path = project_path_base + "/data/templates.json"

    if args.d is None:
        DEVICE = 0
    else:
        DEVICE = args.d
    if args.BATCH is None:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = args.BATCH

    MAX_LENGTH = 200  # changed from 50 for debugging purposes
    MAX_NEW_TOKENS = 1
    softmax = torch.nn.Softmax(dim=-1)
    SEP_BY_AXIS = False
    DEBUG = True
    model_record_df = []

    ## model init
    if not ACCELERATE:
        if "t5" in model:
            model_obj = (AutoModelForSeq2SeqLM.from_pretrained(model))
        elif "llama" in model:
            if LORA:
                model_obj = LlamaForCausalLM.from_pretrained('/shared/4/models/llama2/pytorch-versions/llama-2-7b/')
                peft_model_id = model
                config = PeftConfig.from_pretrained(peft_model_id)
                model_obj = PeftModel.from_pretrained(model_obj, peft_model_id)
            else:
                model_obj = (LlamaForCausalLM.from_pretrained(model))
        else:
            model_obj = (AutoModelForCausalLM.from_pretrained(model))
        model_obj.to(DEVICE)
    
    else:
        '''
        config = AutoConfig.from_pretrained(model)
        with init_empty_weights():
            if "t5" in model:
                model_obj = (AutoModelForSeq2SeqLM.from_config(config))
            #elif "llama" in model:
            #    model_obj = (LlamaForCausalLM.from_config(config))
            else:
                model_obj = (AutoModelForCausalLM.from_config(config))

        model_obj.tie_weights()
        model_obj = load_checkpoint_and_dispatch(
            model_obj, model, device_map="auto"
        )
        '''
        if "t5" in model:
            model_obj = AutoModelForSeq2SeqLM.from_pretrained(model, device_map='auto', cache_dir="/shared/4/models")
        else:
            model_obj = AutoModelForCausalLM.from_pretrained(model, 
                                            #load_in_8bit=True, 
                                            cache_dir="/shared/4/models",
                                            device_map='auto')

    if 'llama' in model:
        tokenizer = LlamaTokenizer.from_pretrained(model,
                                                   truncation=True,
                                                   padding=True,
                                                   padding_side='left',
                                                   max_length=512,
                                                   model_max_length=512
                                                   )
    elif (('t5' in model) and ('4chan' in model)):
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large",
                                                  truncation=True,
                                                  padding=True,
                                                  padding_side='left',
                                                  max_length=512,
                                                  model_max_length=512,
                                                  cache_dir="/shared/4/models")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model,
                                                  truncation=True,
                                                  padding=True,
                                                  padding_side='left',
                                                  max_length=512,
                                                  model_max_length=512,
                                                  cache_dir="/shared/4/models"
                                                  )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ###

    counter = 1
    for axis in SCALE_LIST:
        model_record_df = []
        print("on instrument " + str(counter) + " of " + str(len(SCALE_LIST)))

        prompt_path = project_path_base + "/data/paraphrased-prompts-modified/" + axis + ".json"
        t_prompt_text, j_prompt_text = load_prompts(prompt_path)

        with open(templates_path) as f:
            templates_dict = json.load(f)
            truth_adj_list = templates_dict["truth_options"]
            judgment_adj_list = templates_dict["judgment_options"]
            t_pos_options = [t[0] for t in truth_adj_list]
            t_neg_options = [t[1] for t in truth_adj_list]
            j_pos_options = [t[0] for t in judgment_adj_list]
            j_neg_options = [t[1] for t in judgment_adj_list]

        print("CUDA status: " + str(torch.cuda.is_available()))
        print("number of prompts: " + str(len(t_prompt_text) + len(j_prompt_text)))

        truth_batched_instances = [t_prompt_text[x:(x + BATCH_SIZE)] for x in range(0, len(t_prompt_text), BATCH_SIZE)]
        judgment_batched_instances = [j_prompt_text[x:(x + BATCH_SIZE)] for x in range(0, len(j_prompt_text), BATCH_SIZE)]

        truth_adj_list = flatten(truth_adj_list)
        judgment_adj_list = flatten(judgment_adj_list)

        extras = [' Yes', ' No', ' True', ' False', ' yes', ' no', ' true', ' false', 'Yes', 'No', 'True', 'False']

        # quirks of tokenization
        if model.startswith('facebook'):
            truth_force_words_ids = [f[1] for f in tokenizer(truth_adj_list + extras).input_ids]
            judgment_force_words_ids = [f[1] for f in tokenizer(judgment_adj_list).input_ids]
        elif model.startswith('xlnet'):
            truth_force_words_ids = [f[0] for f in tokenizer(truth_adj_list + extras).input_ids]
            judgment_force_words_ids = [f[0] for f in tokenizer(judgment_adj_list).input_ids]
        else:
            truth_force_words_ids = tokenizer(truth_adj_list + extras).input_ids
            judgment_force_words_ids = tokenizer(judgment_adj_list).input_ids

        truth_probs = evaluate_with_prompts_causal(truth_batched_instances,
                                                   truth_force_words_ids,
                                                   model_obj,
                                                   tokenizer,
                                                   SEQ2SEQ=(model in seq2seq))

        judgment_probs = evaluate_with_prompts_causal(judgment_batched_instances,
                                                      judgment_force_words_ids,
                                                      model_obj,
                                                      tokenizer,
                                                      SEQ2SEQ=(model in seq2seq))

        probs = truth_probs
        probs.update(judgment_probs)
        temp = []

        for prompt, top_probs in probs.items():
            result_dict = {'prompt': prompt}
            for token, prob in top_probs.items():
                result_dict["p(" + token + ")"] = prob

            temp.append(result_dict)

        model_record_df += temp
        counter += 1
        print("got predictions")

        model_record_df = pd.DataFrame(model_record_df)

        m = model.replace("/", "-")
        formatted_prefix = formatted_results_path_base + axis + "-" + m
        if LONGER_SEQS:
            formatted_prefix += "_longerseqs"
        formatted_prefix += ".csv"

        model_record_df.to_csv(formatted_prefix)

        print("probabilities dumped")