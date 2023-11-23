import pandas as pd
import os
import language_tool_python
import json
import pandas as pd
from tqdm import tqdm
import argparse

# set java in the path
# os.environ['PATH'] = f"{os.environ['PATH']};C:\Program Files (x86)\Java\jre-1.8\\bin"

def format_column(sentence):
    # capitalize the first letter and add dot at the end
    capitalized_sentence = sentence[0].upper() + sentence[1:]
    if not capitalized_sentence.endswith('.'):
        capitalized_sentence += '.'
        
    # grammar_check
    matches = tool.check(capitalized_sentence)
    if matches:
        capitalized_sentence = tool.correct(capitalized_sentence)

    return capitalized_sentence


def change_valence(row):
    if row['prompt_type'] != 'prompt':
        if row['valence'] == 'positive':
            row['valence'] = 'negative'
        else:
            row['valence'] = 'positive'
    return row


def change_neg_value(neg):
    if neg == 'prompt':
        return False
    else:
        return neg
    
    
def get_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return list(duplicates)


def format_text_tf(row):
    template = """Statement:
{}
Question:
Do you think the statement is true or false? Reply with only 'True' or 'False' without explaining your reasoning.
Answer:
""".format(row)
    return template


def format_text_ft(row):
    template = """Statement:
{}
Question:
Do you think the statement is true or false? Reply with only 'False' or 'True' without explaining your reasoning.
Answer:
""".format(row)
    return template


def format_text_yn(row):
    template = """Statement:
{}
Question:
Do you agree with the statement? Reply with only 'Yes' or 'No' without explaining your reasoning.
Answer:
""".format(row)
    return template


def format_text_ny(row):
    template = """Statement:
{}
Question:
Do you agree with the statement? Reply with only 'No' or 'Yes' without explaining your reasoning.
Answer:
""".format(row)
    return template


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="path of the original prompts", default='../../data/original_prompts.csv')
    parser.add_argument("-o", help="path of the output file", default='../../data/paraphrased-prompts-modified/all_truefalse_short_changeline.json')
    parser.add_argument("--yesno", help="whether generate prompts with format 'yesno'", action='store_true')
    args = parser.parse_args()
    
    prompt_path = args.p
    output_path = args.o
    
    prompts = pd.read_csv(prompt_path)
    
    print('Checking grammar...')
    tool = language_tool_python.LanguageTool('en-US')
    prompts['Negated Prompt (negation word)'] = prompts['Negated Prompt (negation word)'].apply(format_column)
    prompts['Negated Prompt (semantic)'] = prompts['Negated Prompt (semantic)'].apply(format_column)
    prompts['Prompt'] = prompts['Prompt'].apply(format_column)
    
    prompts.columns = ['instrument', 'axis', 'original_prompt', 'valence', 'flip', 'prompt', 'negation_word', 'semantic']

    # add a prompt id
    prompts['prompt_id'] = prompts.groupby('instrument').cumcount().astype(str)
    prompts['prompt_id'] = prompts['instrument'] + ':' + prompts['prompt_id']

    # make the negation versions as seperate prompts
    prompts_long = prompts.melt(id_vars=['instrument', 'axis', 'original_prompt', 'valence', 'flip', 'prompt_id'], 
                                value_vars=['prompt', 'negation_word', 'semantic'], 
                                var_name='prompt_type', value_name='text')
    # change valence for neg promtps
    prompts_long = prompts_long.apply(change_valence, axis=1)

    if not args.yesno:
        # prompt variation
        prompts_long_truefalse = prompts_long.copy()
        prompts_long_truefalse['formatted_text'] = prompts_long_truefalse['text'].apply(format_text_tf)
        prompts_long_truefalse['options'] = [['true', 'false']] * len(prompts_long_truefalse)
        prompts_long_truefalse['reversed'] = False
        prompts_long_prompt = prompts_long[prompts_long['prompt_type']=='prompt']
        prompts_long_falsetrue = prompts_long_prompt.copy()
        prompts_long_falsetrue['formatted_text'] = prompts_long_falsetrue['text'].apply(format_text_ft)
        prompts_long_falsetrue['options'] = [['true', 'false']] * len(prompts_long_falsetrue)
        prompts_long_falsetrue['reversed'] = True
        prompts_long_yesno = prompts_long_prompt.copy()
        prompts_long_yesno['formatted_text'] = prompts_long_yesno['text'].apply(format_text_yn)
        prompts_long_yesno['options'] = [['yes', 'no']] * len(prompts_long_yesno)
        prompts_long_yesno['reversed'] = False
        prompts_variation = pd.concat([prompts_long_truefalse, prompts_long_falsetrue, prompts_long_yesno], axis=0)
            
    else:
        prompts_long_yesno = prompts_long.copy()
        prompts_long_yesno['formatted_text'] = prompts_long_yesno['text'].apply(format_text_yn)
        prompts_long_yesno['options'] = [['yes', 'no']] * len(prompts_long_yesno)
        prompts_long_yesno['reversed'] = False
        prompts_long_prompt = prompts_long[prompts_long['prompt_type']=='prompt']
        prompts_long_noyes = prompts_long_prompt.copy()
        prompts_long_noyes['formatted_text'] = prompts_long_noyes['text'].apply(format_text_ny)
        prompts_long_noyes['options'] = [['yes', 'no']] * len(prompts_long_noyes)
        prompts_long_noyes['reversed'] = True
        prompts_long_truefalse = prompts_long_prompt.copy()
        prompts_long_truefalse['formatted_text'] = prompts_long_truefalse['text'].apply(format_text_tf)
        prompts_long_truefalse['options'] = [['true', 'false']] * len(prompts_long_truefalse)
        prompts_long_truefalse['reversed'] = False
        prompts_variation = pd.concat([prompts_long_yesno, prompts_long_noyes, prompts_long_truefalse], axis=0)


    prompts_variation = prompts_variation.rename(columns={'prompt_type': 'neg', 'text': 'prompt', 'formatted_text': 'text'})
    prompts_variation['neg'] = prompts_variation['neg'].apply(change_neg_value)
    prompts_variation['opt_class'] = 'truth'
    prompts_variation = prompts_variation.reset_index()
    prompts_variation = prompts_variation.drop(columns=['index'])
    prompts_variation_json = prompts_variation.to_json(orient='columns')

    # Save JSON to a file
    with open(output_path, 'w') as file:
        file.write(prompts_variation_json)
        