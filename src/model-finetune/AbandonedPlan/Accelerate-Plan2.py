from accelerate import Accelerator
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

# Initialize the accelerator
accelerator = Accelerator()

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

optimizer = AdamW(model.parameters(), lr=3e-5)

# Load the dataset and create a dataloader
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


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
        # Tokenize the input text
        input_text = batch["text"]
        #input_ids = tokenizer(input_text, return_tensors="pt", padding=True)["input_ids"]
        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)["input_ids"]

        # Move the input to the device
        input_ids = accelerator.prepare(input_ids)
        
        #outputs = model(**batch)
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)