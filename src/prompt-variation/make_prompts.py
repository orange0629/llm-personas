import argparse
import json
import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import random
# nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('english')


def load_prompts(template_file, prompt_file, ADD_MASK=False, FUZZ_ARGS=False):
    with open(prompt_file) as f:
        prompt_data = json.load(f)

    prompt_class = prompt_data["class"]
    value_dict = prompt_data["value_key"]  # should be set of axes pointing to positive and negative items

    prompt_text_list = []  # will be set of dictionaries, each one prompt
    with open(template_file) as f:
        template_dict = json.load(f)
        templates = template_dict["templates"][prompt_class]  # should be truth and judgment

        subj_nouns = template_dict["subjects"]
        obj_nouns = template_dict["objects"]
        hoopers = template_dict["hooper"]
        verb_endings = template_dict["verb_endings"]
        copulas = template_dict["copulas"]

        for opt_class in ["truth", "judgment"]:
            opt_set = template_dict[opt_class + "_options"]  # should be list of strings...
            if opt_class == "truth":
                interrog_set = template_dict["interrogatives"]
            else:
                interrog_set = [None]

            for opt in opt_set:
                for interrogative in interrog_set:
                    for reverse_prompt in [True, False]:
                        for at_end in [True, False]:
                            for t in templates[opt_class]:
                                if "<neg>" in t:
                                    neg_set = [True, False]
                                else:
                                    neg_set = [None]

                                if "<hooper>" in t:
                                    hooper_set = hoopers
                                else:
                                    hooper_set = [None]

                                for neg in neg_set:
                                    for hooper in hooper_set:
                                        if ADD_MASK:
                                            t += "<mask>"

                                        for axis in value_dict.keys():
                                            for valence in value_dict[axis].keys():
                                                values = value_dict[axis][valence]

                                                for value in values:
                                                    if "<obj>" in t:
                                                        for o in obj_nouns:
                                                            prompt = fill_template(t, value, opt, interrogative, reverse_prompt, at_end, neg, opt_class, obj=o, hooper=hooper, ve=verb_endings, cop=copulas)
                                                            prompt_obj = {"text": prompt, "template": t, "agent": o,
                                                                          "person": "obj", "args": value, "options": opt,
                                                                          "interrog": interrogative, "reversed": reverse_prompt,
                                                                          "at_end": at_end, "neg": neg, "valence": valence,
                                                                          "axis": axis, "hooper": hooper, "opt_class": opt_class}
                                                            prompt_text_list.append(prompt_obj)

                                                    elif "<subj>" in t:
                                                        for s in subj_nouns:
                                                            prompt = fill_template(t, value, opt, interrogative, reverse_prompt, at_end, neg, opt_class, subj=s, hooper=hooper, ve=verb_endings, cop=copulas)
                                                            prompt_obj = {"text": prompt, "template": t, "agent": s,
                                                                          "person": "subj", "args": value, "options": opt,
                                                                          "interrog": interrogative, "reversed": reverse_prompt,
                                                                          "at_end": at_end, "neg": neg, "valence": valence,
                                                                          "axis": axis, "hooper": hooper, "opt_class": opt_class}
                                                            prompt_text_list.append(prompt_obj)

                                                    else:
                                                        prompt = fill_template(t, value, opt, interrogative, reverse_prompt, at_end, neg, opt_class, hooper=hooper, ve=verb_endings, cop=copulas)
                                                        prompt_obj = {"text": prompt, "template": t, "agent": None,
                                                                      "person": None, "args": value, "options": opt,
                                                                      "interrog": interrogative, "reversed": reverse_prompt,
                                                                      "at_end": at_end, "neg": neg, "valence": valence,
                                                                      "axis": axis, "hooper": hooper, "opt_class": opt_class}
                                                        prompt_text_list.append(prompt_obj)

        print("loaded and filtered dataset")

    return pd.DataFrame(prompt_text_list)


def fill_template(prompt: str, value: str, opt: [str, str], interrogative: str, reverse_opts: bool, at_end: bool, neg: bool, opt_class: str, subj=None, obj=None, NEG_PARTICLE="not ", hooper=None, ve=None, cop=None):
    prompt = prompt.replace("<arg>", value)  # required

    if subj is not None:
        assert("<subj>" in prompt)
        prompt = prompt.replace("<subj>", subj)

        if "<cop>" in prompt:
            prompt = prompt.replace("<cop>", cop[subj])

    if obj is not None:
        assert("<obj>" in prompt)
        prompt = prompt.replace("<obj>", obj)

    if opt_class == "truth":
        prompt = add_options(at_end, interrogative, opt, prompt, reverse_opts)

    if neg is not None:
        if neg:
            prompt = prompt.replace("<neg>", NEG_PARTICLE)
        else:
            prompt = prompt.replace("<neg>", "")

    if hooper is not None:
        assert ("<hooper>" in prompt)

        if "<hooper>ing" in prompt:
            token = WordNetLemmatizer().lemmatize(hooper)
            prompt = prompt.replace("<hooper>ing", token + "ing")  # hope this works
        elif "<hooper><end>" in prompt:
            prompt = prompt.replace("<hooper>", hooper)
            assert(subj is not None)
            prompt = prompt.replace("<end>", ve[subj])
        else:
            prompt = prompt.replace("<hooper>", hooper)

    return prompt


def add_options(at_end, interrogative, opt, prompt, reverse_prompt):
    if opt is not None:
        assert ("<opt>" in prompt)
        assert ("<interrogative>" in prompt)
        prompt = prompt.replace("<opt><interrogative>", "")
        if reverse_prompt:
            opt_string = opt[1] + " or " + opt[0]
        else:
            opt_string = opt[0] + " or " + opt[1]
        opt_string = opt_string.capitalize()
        prompt = prompt.capitalize()

        if at_end:
            prompt = prompt + " " + opt_string + interrogative
        else:
            prompt = opt_string + interrogative + " " + prompt
    return prompt


def fuzz_arg(prompt_obj, value, MAX_FUZZ=2):
    fuzzed_prompt_list = []
    tokenized_value = nltk.tokenize.word_tokenize(value, language='english', preserve_line=False)
    content_tokens = [v for v in tokenized_value if v not in STOPWORDS]
    chosen = random.sample(content_tokens, MAX_FUZZ)
    assert(MAX_FUZZ < len(chosen))

    for i in range(MAX_FUZZ):
        synonym = wordnet.synset(chosen[i])
        new_prompt = prompt_obj["prompt"].replace(chosen, synonym)
        new_prompt_obj = prompt_obj.copy()
        new_prompt_obj["prompt"] = new_prompt

        fuzzed_prompt_list.append(new_prompt_obj)

    return fuzzed_prompt_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="template input file")
    parser.add_argument("-p", help="prompt input file")
    parser.add_argument("-d", help="output file name")
    parser.add_argument("--mask", help="add mask to prompts", action='store_true')
    args = parser.parse_args()

    if args.f is None:
        print("No template input file given!")

    if args.p is None:
        print("No prompt input file given!")

    if args.d is None:
        args.d = "/home/laviniad/projects/IdeologyProbes/src/probes/prompt_data/fuzzed_surveys/" + "base.json"

    template_json_name = args.f
    prompt_json_name = args.p
    output_file = args.d
    prompt_list_name = "prompt_list"

    prompt_df = load_prompts(template_file=template_json_name, prompt_file=prompt_json_name, ADD_MASK=args.mask)

    with open(output_file, 'w') as out_f:
        prompt_df.to_json(out_f)
        print("prompt df dumped")
