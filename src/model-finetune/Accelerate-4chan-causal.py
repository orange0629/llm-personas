from accelerate import Accelerator
from transformers import AdamW, AutoTokenizer, AutoModelForCausalLM, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
import datasets
import transformers
from finetune_utils.utils import get_4chan_dataset, get_bible_dataset

batch_size = 4
num_epochs = 3
max_sentence_len = 512
model_name = "/shared/4/models/llama2/pytorch-versions/llama-2-7b/"
output_dir = "/shared/2/projects/llm-personas/llama2-4chan-linear/"

# Initialize the accelerator
accelerator = Accelerator()

# Initialize TensorBoard Summary Writer
summary_writer = SummaryWriter()

# To have only one message (and not 8) per logs of Transformers or Datasets, we set the logging verbosity
# to INFO for the main process only.
if accelerator.is_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

# Load the tokenizer and model
if "llama" in model_name:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)


tokenized_dataset = get_4chan_dataset(tokenizer, max_sentence_len, need_labels=True)
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=3e-5)

# Move the model to the device
dataloader, model, optimizer = accelerator.prepare(
    dataloader, model, optimizer
)


num_training_steps = num_epochs * len(dataloader)
num_training_tokens = num_epochs * len(dataloader) * batch_size * max_sentence_len
print("You will be training on", num_training_tokens, "tokens in total.")

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
model.train()


# Generate 10 Checkpoints linearly in the training steps
checkpoints = list(range(0, num_training_tokens, num_training_tokens // 11))[1:]


for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        outputs = model(input_ids=batch['input_ids'], labels=batch['labels'])
        loss = outputs.loss
        summary_writer.add_scalar('Loss', loss, (step + epoch * len(dataloader)) * batch_size * max_sentence_len)
        accelerator.backward(loss)
    
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # Save Checkpoints in the list defined above
        if (len(checkpoints) != 0) and ((step + epoch * len(dataloader)) * batch_size * max_sentence_len >= checkpoints[0]):
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir+"checkpoint-"+str(checkpoints[0]), 
                                            is_main_process=accelerator.is_main_process,
                                            save_function=accelerator.save,
                                            state_dict=accelerator.get_state_dict(model),)
            checkpoints.pop(0)

# Save Final Model
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(output_dir+"checkpoint-final", 
                                is_main_process=accelerator.is_main_process,
                                save_function=accelerator.save,
                                state_dict=accelerator.get_state_dict(model),)
