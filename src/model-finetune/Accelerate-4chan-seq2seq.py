from accelerate import Accelerator
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from transformers import BartTokenizer, BartModel, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
from finetune_utils.utils import get_4chan_dataset, get_bible_dataset

batch_size = 4
max_sentence_len = 512
output_dir = "/shared/2/projects/llm-personas/flan-t5-large-4chan/"

# Initialize the accelerator
# accelerator = Accelerator()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

tokenized_dataset = get_4chan_dataset(tokenizer, max_sentence_len, need_labels=True)


args = Seq2SeqTrainingArguments(
    output_dir, 
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    #fp16=True #available only with CUDA
    )


#trainer = accelerator.prepare(Seq2SeqTrainer(
trainer = (Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)
)

trainer.train()

model.save_pretrained(output_dir+"checkpoint-final")

