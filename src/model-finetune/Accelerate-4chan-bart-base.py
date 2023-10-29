from accelerate import Accelerator
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from transformers import BartTokenizer, BartModel, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import datasets
import transformers
import torch

batch_size = 2
max_sentence_len = 512
output_dir = "/shared/2/projects/llm-personas/bart-base-4chan/"

# Initialize the accelerator
accelerator = Accelerator()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")


f1 = open('4chan_dataset/pol_clean_noline.txt')
s = f1.read()
f1.close()

tokened = tokenizer(s)
#new_dict = {'input_ids': [], 'attention_mask': [], 'decoder_input_ids': [], 'decoder_attention_mask': [], 'labels': []}
new_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}
for i in range(0, len(tokened['input_ids']), max_sentence_len):
    if(i + max_sentence_len >= len(tokened['input_ids'])):
        break
    new_dict['input_ids'].append(tokened['input_ids'][i:(i+max_sentence_len)])
    new_dict['attention_mask'].append(tokened['attention_mask'][i:(i+max_sentence_len)])
    #new_dict['decoder_input_ids'].append(tokened['input_ids'][i:(i+max_sentence_len)])
    #new_dict['decoder_attention_mask'].append(tokened['attention_mask'][i:(i+max_sentence_len)])
    new_dict['labels'].append(tokened['input_ids'][i:(i+max_sentence_len)])
tokenized_dataset = Dataset.from_dict(new_dict)
tokenized_dataset.set_format("torch")

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")


args = Seq2SeqTrainingArguments(
    output_dir, 
    learning_rate=3e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True #available only with CUDA
    )


trainer = accelerator.prepare(Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
)

trainer.train()

model.save_pretrained(output_dir+"checkpoint-final")

