from datasets import Dataset

data_dir_4chan = '4chan_dataset/pol_clean_noline.txt'
data_dir_bible = 'bible_dataset/bible_clean_noline.txt'

def get_4chan_dataset(tokenizer, max_sentence_len = 512, need_labels = True):
    return get_custom_dataset(tokenizer, data_dir_4chan, max_sentence_len, need_labels)

def get_bible_dataset(tokenizer, max_sentence_len = 512, need_labels = True):
    return get_custom_dataset(tokenizer, data_dir_bible, max_sentence_len, need_labels)


def get_custom_dataset(tokenizer, text_dir, max_sentence_len = 512, need_labels = True):
    f1 = open(text_dir)
    s = f1.read()
    f1.close()

    tokened = tokenizer(s)
    #new_dict = {'input_ids': [], 'attention_mask': [], 'decoder_input_ids': [], 'decoder_attention_mask': [], 'labels': []}
    if(need_labels):
        new_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}
    else:
        new_dict = {'input_ids': [], 'attention_mask': []}
    
    for i in range(0, len(tokened['input_ids']), max_sentence_len):
        if(i + max_sentence_len >= len(tokened['input_ids'])):
            break
        new_dict['input_ids'].append(tokened['input_ids'][i:(i+max_sentence_len)])
        new_dict['attention_mask'].append(tokened['attention_mask'][i:(i+max_sentence_len)])
        if(need_labels):
            new_dict['labels'].append(tokened['input_ids'][i:(i+max_sentence_len)])
    tokenized_dataset = Dataset.from_dict(new_dict)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset