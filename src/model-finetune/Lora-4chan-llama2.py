from accelerate import Accelerator
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, TrainingArguments
import transformers
from tqdm import tqdm
from finetune_utils.utils import get_4chan_dataset
from finetune_utils.utils import print_trainable_parameters
import torch

batch_size = 2
max_sentence_len = 512
output_dir = "/shared/2/projects/llm-personas/bloomz-3b-4chan-lora/"


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b", 
                                         #load_in_8bit=True, 
                                         device_map='auto')


tokenized_dataset = get_4chan_dataset(tokenizer, max_sentence_len, need_labels=True)

for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


args = TrainingArguments(
    output_dir, 
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    #weight_decay=0.01,
    #save_total_limit=2,
    save_steps=100,
    num_train_epochs=3,
    #predict_with_generate=True,
    fp16=True #available only with CUDA
    )


trainer = transformers.Trainer(
    model, 
    args,
    train_dataset=tokenized_dataset,
)

trainer.train()

model.save_pretrained(output_dir+"checkpoint-final")

