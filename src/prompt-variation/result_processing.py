import pandas as pd
import os
import json
import re
import argparse


# Function to create 'positive' and 'negative' columns based on 'options'
def create_pos_neg_by_valence(row):
    if row['valence'] == 'positive':
        return row['positive_max'], row['negative_max']
    elif row['valence'] == 'negative':
        return row['negative_max'], row['positive_max']
    
    
def create_positive_negative(row):
    options = row['options']
    try:
        positive_values = row[options[0]].max()
    except:
        positive_values = row[options[0]]
    try:
        negative_values = row[options[1]].max()
    except:
        negative_values = row[options[1]]
    return positive_values, negative_values


def if_agree(row):
    if row['positive_max'] > row['negative_max']:
        return True
    else:
        return False
    
    
def if_personas(row):
    if row['positive_max_valence'] > row['negative_max_valence']:
        return True
    else:
        return False
    

def create_df(model_result: pd.DataFrame, prompt_df: pd.DataFrame):
    target_columns = ['prompt']
    option_columns = ['true', 'false', 'yes', 'no', 'good', 'bad']
    
    columns_to_keep = list(prompt_df.columns)

    target_columns = target_columns + option_columns
    
    # formalized the column names, delete the 'p()' and lowercase them
    print('Formalizing the column names...')
    model_result.columns = [re.sub(r'^p\(\s*(.*?)\s*\)$', r'\1', col).lower() if col.startswith('p(') else col.lower() for col in model_result.columns]
    
    # only keep the columns that are yes/no related
    common_columns = set(model_result.columns).intersection(set(target_columns))
    model_result = model_result.loc[:, list(common_columns)]

    # merge the two dataframes
    model_result['text'] = model_result['prompt']
    model_result = model_result.drop(columns=['prompt'])

    combined = prompt_df.merge(model_result, how='left', on='text')

    # Create 'positive' and 'negative' columns based on the 'options'
    print('Creating the positive, negative columns...')
    combined[['positive_max', 'negative_max']] = combined.apply(create_positive_negative, axis=1, result_type='expand')
    
    columns_to_keep += ['positive_max', 'negative_max']
    combined = combined[columns_to_keep]

    combined[['positive_max_valence', 'negative_max_valence']] = combined.apply(create_pos_neg_by_valence, axis=1, result_type='expand')
    
    combined['agree'] = combined.apply(if_agree, axis=1, result_type='expand')
    combined['agree_personas'] = combined.apply(if_personas, axis=1, result_type='expand')
    
    return combined


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help="path of the raw results directory", default='../../result/all_truefalse_short_changeline/')
    parser.add_argument("-p", help="path of the prompts", default='../../data/paraphrased-prompts-modified/all_truefalse_short_changeline.json')
    parser.add_argument("-o", help="output directory", default="../../result/")
    args = parser.parse_args()
    
    target_path = args.r
    prompt_path = args.p
    output_path = args.o

    # Initialize an empty DataFrame to hold concatenated results
    all_results_df = pd.DataFrame()

    files = os.listdir(target_path)

    for file in files:
        if file.startswith('.'):
            continue
        print(file)
        result = pd.read_csv(os.path.join(target_path, file)).drop(columns='Unnamed: 0')

        with open(prompt_path, "r") as json_file:
            prompts = json.load(json_file)

        prompt_df = pd.DataFrame(prompts)

        df_result = create_df(result, prompt_df)

        if 'llama' in file:
            modelname = file.split('versions-')[1].split('.')[0]
        elif 'bloomz' in file:
            modelname = file.split('bigscience-')[1].split('.')[0]
        elif 'gpt' in file:
            modelname = 'gpt2'
        elif 'falcon' in file:
            modelname = 'falcon-7b'
        elif 'Pajama' in file:
            modelname = 'redpajamas-incite-7b-instruct'
        elif 't5' in file:
            modelname = file.split('google-')[1].split('.')[0]
        else:
            modelname = file

        # Concatenate the current df_result to the all_results_df
        df_result['model'] = modelname
        all_results_df = pd.concat([all_results_df, df_result], ignore_index=True)

        # Save individual df_result
        print('Dataframe Created:', file)

    # After the loop, you can save the concatenated DataFrame as well
    all_results_df.to_csv(output_path+target_path.split('/')[3]+'.csv', index=False)
    print('All Dataframes Concatenated')

