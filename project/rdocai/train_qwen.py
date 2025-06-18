import json

from tqdm.auto import tqdm

import torch

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from qwen_vl_utils import process_vision_info

def make_chat(sample):
    question = "Please return the contents of this receipt image as structured JSON data."
    chat = []
    
    gt_json = json.dumps(json.loads(sample["ground_truth"])["gt_parse"], ensure_ascii=False)
    img = sample["image"].convert("RGB").resize((224, 224))
    
    chat.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", "text": question},
            ],
        }
    )
    chat.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "```json\n" + gt_json + "\n```",
                }
            ],
        }
    )

    return chat

def collate_fn(examples):

    texts = [processor.apply_chat_template(make_chat(sample), tokenize=False, add_generation_prompt=False).strip() for sample in examples]
    image_inputs = [process_vision_info(make_chat(sample))[0] for sample in examples]
    
    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(tok) for tok in [
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>",
        ]
    ]
    # Mask tokens for not being used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch  # Return the prepared batch

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id, 
    # device_map="auto", 
    torch_dtype=torch.bfloat16,
    # quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "right"

# Configure QLoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

training_args = SFTConfig(
    output_dir=f"{model_id}-sft-llava-instruct-mix-vsft",     # Directory to save the model and push to the Hub. Use a specific repository id (e.g., gemma-3-4b-it-trl-sft-MMIU-Benchmark for multi-image datasets).
    num_train_epochs=3,                                             # Set the number of epochs to train the model.
    per_device_train_batch_size=4,                                  # Batch size for each device (e.g., GPU) during training. multi-image -> per_device_train_batch_size=1
    gradient_accumulation_steps=4,                                  # Number of steps before performing a backward/update pass to accumulate gradients. multi-image -> gradient_accumulation_steps=1
    gradient_checkpointing=True,                                    # Enable gradient checkpointing to reduce memory usage during training.
    optim="adamw_torch_fused",                                      # Use the fused AdamW optimizer for better performance.
    logging_steps=5,                                               # Frequency of logging training progress (log every 10 steps).
    save_strategy="epoch",                                          # Save checkpoints at the end of each epoch.
    learning_rate=1e-04,                                            # Learning rate for training.
    bf16=True,                                                      # Enable bfloat16 precision for training to save memory and speed up computations.
    push_to_hub=False,                                               # Automatically push the fine-tuned model to Hugging Face Hub after training.
    report_to="wandb",                                        # Automatically report metrics to tensorboard.
    gradient_checkpointing_kwargs={"use_reentrant": False},         # Set gradient checkpointing to non-reentrant to avoid issues.
    dataset_kwargs={"skip_prepare_dataset": True},                  # Skip dataset preparation to handle preprocessing manually.
    remove_unused_columns=False,                                    # Ensure unused columns are not removed in the collator (important for batch processing).
)

# Load the dataset
# ds = load_dataset("naver-clova-ix/cord-v2", split="train")
ds = load_dataset("doolayer/cord-v2-rotated-cut", split="train")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=ds,
    processing_class=processor,
    peft_config=peft_config,
)

trainer.train()

# Save the final model
trainer.save_model()