from accelerate import Accelerator
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import datasets
import transformers

# Initialize the accelerator
accelerator = Accelerator()

# To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
# to INFO for the main process only.
if accelerator.is_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Define the Tokenize function
def tokenize_function(examples):
    outputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    #outputs = tokenizer(examples["text"], return_tensors="pt", padding=True)
    return outputs

# Load the dataset and create a dataloader
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch")

model = AutoModelForCausalLM.from_pretrained("gpt2")

    
optimizer = AdamW(model.parameters(), lr=3e-5)

dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# Move the model to the device
dataloader, model, optimizer = accelerator.prepare(
    dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
model.train()
    
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch['input_ids'], labels=batch['input_ids'])
        loss = outputs.loss
        accelerator.backward(loss)
    
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)