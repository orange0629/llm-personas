import json
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers import BartForConditionalGeneration
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map  # somewhat experimental

project_path_base = "F:/david/llm-personas"

CAUSAL = False
causal_models = ["bigscience/bloom-1b1",
"EleutherAI/gpt-neo-1.3B",
"Salesforce/codegen-350M-mono",
"microsoft/prophetnet-large-uncased",
"facebook/bart-large",
"facebook/xglm-564M",
"xlnet-large-cased",
"bigscience/bloomz-560m",
"facebook/opt-350m"]
mask_models = ["distilroberta-base", "roberta-base", "roberta-large", "distilbert-base-cased", "bert-base-cased",
               "bert-large-cased", "bert-base-multilingual-cased"]

mask_models += ["cardiffnlp/twitter-roberta-base",
                "nlpaueb/legal-bert-base-uncased",
                "GroNLP/hateBERT",
                "microsoft/codebert-base-mlm"]

abbrev_models = ["roberta-base", "roberta-large", "distilbert-base-cased", "bert-base-multilingual-cased"]
seq2seq = ["t5-small", "t5-base", "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"]

MODEL_TYPES = {"roberta": ['distilroberta-base', 'roberta-base', 'roberta-large', 'cardiffnlp/twitter-roberta-base',
                           'microsoft/codebert-base-mlm', 'cardiffnlp/twitter-roberta-base-jun2020',
                           'cardiffnlp/twitter-roberta-base-jun2021', 'cardiffnlp/twitter-roberta-base-jun2022'],
               "bert": ["distilbert-base-cased", "bert-base-cased", "bert-large-cased", "bert-base-multilingual-cased",
                        "nlpaueb/legal-bert-base-uncased", "GroNLP/hateBERT"]}

SCALE_LIST = ['OCEAN', 'MFT', 'RVS', 'SIBS', 'HCS', 'UAS', 'RCBS', 'ACI', 'EPQ', 'BSCTM',
'BSSS', 'SBI', 'PDBS', 'BCQ', 'CCS', 'AIS', 'BRS', 'MHBS', 'ONBGS', 'IS', 'ERUS', 'PSNS',
'MAS', 'ATPLS', 'FIS', 'MAQ', 'NBI', 'SSIS', 'MMMS', 'VES', 'DAI', 'UWS', 'PES', 'RIQ', 'TLS']


def load_prompts(prompt_path):
    truth_prompt_text_list = []
    judgment_prompt_text_list = []
    p2template = {}
    p2args = {}
    p2person = {}
    p2options = {}
    p2class = {}
    with open(prompt_path) as f:
        prompt_dict = pd.read_json(f)

        for prompt_row in prompt_dict.itertuples():
            text = prompt_row.text
            if not CAUSAL:
                text = text.strip()  # some values or templates have trailing spaces
                text += " <mask>"
                if ADD_PERIOD:
                    text += "."

            p2template[text] = prompt_row.template
            p2person[text] = prompt_row.person
            p2args[text] = prompt_row.args
            p2options[text] = prompt_row.options
            p2class[text] = prompt_row.opt_class

            if p2class[text] == "truth":
                truth_prompt_text_list.append(text)
            else:
                judgment_prompt_text_list.append(text)

    return truth_prompt_text_list, judgment_prompt_text_list, p2args, p2person, p2template, p2options, p2class


def flatten(l):
    return [item for sublist in l for item in sublist]


def evaluate_with_prompts_causal(batched_instances, adj_list, word_ids, model, tokenizer, prompt_to_options,
                                 pos_options, neg_options, SEQ2SEQ=False):
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
            generated = model.generate(encoded.input_ids,
                                       pad_token_id=tokenizer.pad_token_id,
                                       attention_mask=encoded.attention_mask,
                                       #no_repeat_ngram_size=1,
                                       return_dict_in_generate=True,
                                       #num_beams=num_beams,
                                       output_scores=True,
                                       remove_invalid_values=True,
                                       #num_return_sequences=num_return_sequences,
                                       renormalize_logits=True,
                                       max_length=MAX_LENGTH
                                       )
            #print(generated)

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
                    top_probs[tokenizer.decode(id)] = float(next_token_logits[id].cpu().numpy())
                for tok, id in target_dict.items():
                    # get the probability of the next token id
                    prob = next_token_logits[id]
                    if "\n" in tok:
                        top_probs[tok.replace("\n", "\\n")] = float(prob.cpu().numpy())
                    else:
                        top_probs[tok] = float(prob.cpu().numpy())

            # make the probability binary
            if prompt in prompt_to_options.keys():
                # adj1, adj2 = prompt_to_options[prompt]
                # note: had bug before...
                adj_scores = [0.0, 0.0]
                for tok, id in target_dict.items():
                    if tok in pos_options:
                        adj_scores[0] += float(next_token_logits[id].cpu().numpy())
                    if tok in neg_options:
                        adj_scores[1] += float(next_token_logits[id].cpu().numpy())

                probs[prompt] = {"scores": adj_scores, "token_probs": top_probs}
        count += 1

    return probs


def score_masked_reg():
    pass


def score_masked(b, instance, mask_token_logits, neg_options, pos_options, tokens_of_interest, target_ids):
    prompt = b[instance]
    # context: the mask_token_logits tensor is scores pre-softmax

    # mask_token_probs_soft = softmax(mask_token_logits)

    # use softmax to make them probability
    mask_token_logits = softmax(mask_token_logits)
    mask_token_logits_target = mask_token_logits[:, :, target_ids]
    # NOTE: NOW USING ABSOLUTE DIFFERENCE. taking top 50
    ids = mask_token_logits.topk(k=50).indices.reshape(-1)
    #print("DEBUG" + str(ids))
    top_probs = {}
    for id in ids:
        top_probs[tokenizer.decode(id)] = float(mask_token_logits.squeeze()[id].numpy())
    for id in target_ids:
        top_probs[tokenizer.decode(id)] = float(mask_token_logits.squeeze()[id].numpy())

    if NORM_SCORE:
        rel_probs = mask_token_logits_target[instance, instance]
        norm = np.linalg.norm(rel_probs)
        rel_probs = rel_probs / norm
        adj_scores = [0.0, 0.0]
        for adj in pos_options:
            adj_scores[0] += (rel_probs[tokens_of_interest.index(adj)]).numpy()
        for adj in neg_options:
            adj_scores[1] += (rel_probs[tokens_of_interest.index(adj)]).numpy()
    else:
        rel_probs = mask_token_logits[instance, instance]
        rel_probs = softmax(rel_probs)
        adj_scores = [0.0, 0.0]
        for adj in pos_options:
            adj_scores[0] += (rel_probs[tokens_of_interest.index(adj)]).numpy()
        for adj in neg_options:
            adj_scores[1] += (rel_probs[tokens_of_interest.index(adj)]).numpy()
    score_dict = {"scores": adj_scores, "probs": top_probs}

    return prompt, score_dict


def evaluate_with_prompts_masked(model, batched_instances, tokens_of_interest, pos_options, neg_options):
    probs = {}
    target_ids = [tokenizer.convert_tokens_to_ids(pair) for pair in tokens_of_interest]

    seq_idx = 0
    for b in tqdm(batched_instances):
        b = [p.replace("<mask>", tokenizer.mask_token) for p in b]
        with torch.no_grad():
            encoded = tokenizer(b, max_length=512, truncation=True, padding="max_length", return_tensors='pt').to(
                DEVICE)
            batch_logits = model(**encoded).logits  # should I just be using input_ids?

        encoded = encoded.to('cpu')

        if DEBUG:
            for r in encoded["input_ids"][:]:
                if tokenizer.mask_token_id not in r:
                    print("WARNING: no mask token found in prompt")
        mask_token_index = np.argwhere(encoded["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = batch_logits[:, mask_token_index, :]

        # most_likely_50_batch = []
        # for i in range(len(b)):
        #     m_rn = softmax(batch_logits[i, i, :]) # costly!
        #     most_likely_50_tok = torch.topk(m_rn, 50, dim=0)  # most likely 50!
        #     most_likely_50 = [b[i], most_likely_50_tok.values.tolist(),
        #                       [tokenizer.decode(t) for t in most_likely_50_tok.indices.tolist()]]
        #     most_likely_50_batch.append(most_likely_50)

        mask_token_logits = mask_token_logits.to('cpu')
        for instance in range(len(b)):  # iterate through instances
            # mask_token_logits = batch_logits[:, :, target_ids]
            prompt, score_dict = score_masked(b, instance, mask_token_logits, neg_options, pos_options, tokens_of_interest, target_ids)

            probs[prompt] = score_dict

        seq_idx += 1  # need for counting w/r/t batches!
        # most_likely_50_list += most_likely_50_batch

    return probs #, most_likely_50_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-scale", help="scale name", default="sanity")
    parser.add_argument("-d", help="device", type=int, default=0)
    parser.add_argument("-BATCH", help="device", type=int, default=1)
    parser.add_argument("--causal", help="measure causal, not masked lms", action='store_true')
    parser.add_argument("--normscore", help="use divide by norm, not softmax", action='store_true')
    parser.add_argument("--add_sentence_token", help="add </s> or analogue at end for MLMs", action='store_true')
    parser.add_argument("--add_period", help="add . at end for MLMs", action='store_true')
    parser.add_argument("-model", help="model to evaluate", type=str, default=None)
    parser.add_argument("--accelerate", help="use accelerate", action='store_true')
    parser.add_argument("--record_longer_seqs", help="store sequence scores (instead of just next token probabilities)", action='store_true')
    parser.add_argument("-num_beams", help="number of beams for beam search decoding", type=int, default=10)
    parser.add_argument("-num_returned_seqs", help="number of returned sequences for beam search decoding", type=int, default=1)

    args = parser.parse_args()
    assert (args.num_beams >= args.num_returned_seqs)

    if args.scale is None:
        print("No scale given!")
    # what I add (By Bangzhao)
    else:
        SCALE_LIST = [args.scale]

    CAUSAL = args.causal
    NORM_SCORE = args.normscore

    ADD_S_TOKEN = (not CAUSAL) and args.add_sentence_token
    ADD_PERIOD = (not CAUSAL) and args.add_period
    ACCELERATE = args.accelerate
    LONGER_SEQS = args.record_longer_seqs
    print(args.model)

    if args.model is None:
        print("No model selected!")
    model = args.model

    NUM_BEAMS = args.num_beams
    NUM_SEQS = args.num_returned_seqs

    #formatted_results_path_base = "/shared/3/projects/laviniad/ideologyprobes/prompts/results/bymodel/"
    formatted_results_path_base = project_path_base + "/result/"

    #templates_path = "/home/laviniad/projects/LAMPS/src/probes/prompt_data/templates.json"
    templates_path = project_path_base + "/data/templates.json"

    if args.d is None:
        DEVICE = 0
    else:
        DEVICE = args.d
    if args.BATCH is None:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = args.BATCH

    MAX_LENGTH = 50  # changed from 50 for debugging purposes
    MAX_NEW_TOKENS = 1
    softmax = torch.nn.Softmax(dim=-1)
    SEP_BY_AXIS = False
    DEBUG = True
    model_record_df = []

    ## model init

    if not ACCELERATE:
        if CAUSAL:
            if "t5" in model:
                model_obj = (AutoModelForSeq2SeqLM.from_pretrained(model))#.to(DEVICE)
            elif "bart" in model:
                model_obj = (BartForConditionalGeneration.from_pretrained(model, device_map="auto"))
            else:
                model_obj = (AutoModelForCausalLM.from_pretrained(model))#.to(DEVICE)
        else:
            model_obj = (AutoModelForMaskedLM.from_pretrained(model))#.to(DEVICE)
        model_obj.to(DEVICE)
    else:
        config = AutoConfig.from_pretrained(model)
        with init_empty_weights():
            if "t5" in model:
                model_obj = (AutoModelForSeq2SeqLM.from_config(config))
            else:
                model_obj = (AutoModelForCausalLM.from_config(config))

        model_obj.tie_weights()
        model_obj = load_checkpoint_and_dispatch(
            model_obj, model, device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model,
                                              truncation=True,
                                              padding=True,
                                              padding_side='left',
                                              max_length=512,
                                              model_max_length=512
                                              )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ###

    counter = 1
    for axis in SCALE_LIST:
        print("on instrument " + str(counter) + " of " + str(len(SCALE_LIST)))
        #prompt_path = "/home/laviniad/projects/LAMPS/src/probes/prompt_data/fuzzed_surveys/" + axis + ".json"
        prompt_path = project_path_base + "/data/paraphrased-prompts/" + axis + ".json"
        t_prompt_text, j_prompt_text, prompt_to_topics, prompt_to_person, prompt_to_template, prompt_to_options, prompt_to_class = load_prompts(
            prompt_path)

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

        if not CAUSAL:
            truth_tokens_of_interest = t_pos_options + t_neg_options
            judgment_tokens_of_interest = j_pos_options + j_neg_options

        truth_adj_list = flatten(truth_adj_list)
        judgment_adj_list = flatten(judgment_adj_list)

        if ADD_S_TOKEN and tokenizer.eos_token is not None:
            truth_batched_instances = [[s + tokenizer.eos_token for s in t] for t in truth_batched_instances]
            judgment_batched_instances = [[s + tokenizer.eos_token for s in t] for t in judgment_batched_instances]

        extras = [' Yes', ' No', ' True', ' False',
                  ' yes', ' no', ' true', ' false',
                  'Yes', 'No', 'True', 'False']
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

        if CAUSAL:
            truth_probs = evaluate_with_prompts_causal(truth_batched_instances, truth_adj_list,
                                                       truth_force_words_ids,
                                                       model_obj,
                                                       tokenizer, prompt_to_options, t_pos_options, t_neg_options,
                                                       SEQ2SEQ=(model in seq2seq))
            judgment_probs = evaluate_with_prompts_causal(judgment_batched_instances, judgment_adj_list,
                                                          judgment_force_words_ids,
                                                          model_obj,
                                                          tokenizer, prompt_to_options, j_pos_options, j_neg_options,
                                                          SEQ2SEQ=(model in seq2seq))
        else:
            t_prompt_text = [t.replace("<mask>", tokenizer.mask_token) for t in t_prompt_text]
            j_prompt_text = [t.replace("<mask>", tokenizer.mask_token) for t in j_prompt_text]
            curr_t_pos_options = t_pos_options
            curr_t_neg_options = t_neg_options
            curr_j_pos_options = j_pos_options
            curr_j_neg_options = j_neg_options

            truth_probs = evaluate_with_prompts_masked(model_obj, truth_batched_instances, truth_tokens_of_interest, curr_t_pos_options, curr_t_neg_options)
            judgment_probs = evaluate_with_prompts_masked(model_obj, judgment_batched_instances, judgment_tokens_of_interest, curr_j_pos_options, curr_j_neg_options)

        probs = truth_probs
        probs.update(judgment_probs)
        temp = []
        #adj_score_dict = {}
        for prompt, data in probs.items():
            adj_scores, top_probs = data["scores"], data["token_probs"]
            result_dict = {'instrument': axis, 'prompt': prompt}
            for token, prob in top_probs.items():
                result_dict["p(" + token + ")"] = prob

            temp.append(result_dict)
        #     adj_score_dict[prompt] = adj_scores
        # output_filename = "C:/Users/elain/Desktop/llm-personas-master/result/score_output.json"
        # with open(output_filename, "w") as json_file:
        #     json.dump(adj_score_dict, json_file, indent=4)

        model_record_df += temp
        counter += 1
        print("got predictions")
    model_record_df = pd.DataFrame(model_record_df)

    m = model.replace("/", "-")
    formatted_prefix = formatted_results_path_base + m
    if ADD_PERIOD:
        formatted_prefix += "_period"
    if LONGER_SEQS:
        formatted_prefix += "_longerseqs"
    formatted_prefix += ".csv"

    model_record_df.to_csv(formatted_prefix)

    print("probabilities dumped")
