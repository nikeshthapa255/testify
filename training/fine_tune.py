from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorWithPadding, TrainerCallback
from datasets import load_dataset
import logging
import os
import json
import tempfile
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_function(examples):
    logger.info("Tokenizing examples...")
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Setting the labels to the input_ids
    return tokens

if os.name == 'nt':  # Check if the system is Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the training data
logger.info("Loading training data...")
dataset = load_dataset('text', data_files={'train': 'training_data.txt'}, cache_dir='./dataset_cache')
logger.info(f"Training data loaded with {len(dataset['train'])} examples.")

# Initialize the tokenizer and model
logger.info("Initializing tokenizer and model...")

def load_tokenizer_model(retries=3, output_dir="./results/latest"):
    for attempt in range(retries):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            # Set padding token
            tokenizer.pad_token = tokenizer.eos_token
            if os.path.exists(output_dir):
                model = GPT2LMHeadModel.from_pretrained(output_dir)
                logger.info("Resuming from the last saved model")
            else:
                model = GPT2LMHeadModel.from_pretrained('gpt2')
            return tokenizer, model
        except Exception as e:
            logger.error(f"Error initializing tokenizer or model on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(5)  # Wait before retrying
            else:
                raise e

try:
    tokenizer, model = load_tokenizer_model()
except Exception as e:
    logger.error(f"Failed to initialize tokenizer or model after retries: {e}")
    raise e

logger.info("Tokenizer and model initialized.")

# Tokenize the data
logger.info("Tokenizing the dataset...")
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["text"])
logger.info("Dataset tokenized.")

# Set up data collator with padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set up training arguments
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,  # Adjust based on GPU memory
    num_train_epochs=3,
    save_steps=5000,  # Set to a large number to avoid saving checkpoints
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,  # Increase number of workers for data loading
    logging_dir="./logs",  # Directory for storing logs
    gradient_accumulation_steps=1,  # Accumulate gradients over multiple steps (optional)
    max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
    logging_steps=50,  # Log every 50 steps
)

logger.info("Training arguments set.")

# Custom callback to save model and log metrics at each specified step
class SaveModelStepCallback(TrainerCallback):
    def __init__(self, save_steps=100, tokenizer=None):
        self.save_steps = save_steps
        self.tokenizer = tokenizer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            logger.info(f"Saving model and logging metrics at step {state.global_step}")
            output_dir = os.path.join(args.output_dir, "latest")
            kwargs['model'].save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
            
            # Save logs to a file
            logs_output_file = os.path.join(output_dir, "metrics.json")
            with open(logs_output_file, 'w') as f:
                json.dump(logs, f)

# Create Trainer instance
logger.info("Creating Trainer instance...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    callbacks=[SaveModelStepCallback(save_steps=100, tokenizer=tokenizer)],  # Pass the tokenizer to the callback
)
logger.info("Trainer instance created.")

# Fine-tune the model
logger.info("Starting model fine-tuning...")
trainer.train()
logger.info("Model fine-tuning completed.")

# Save the final model
logger.info("Saving the final fine-tuned model...")
model.save_pretrained("./fine-tuned-gpt2")
tokenizer.save_pretrained("./fine-tuned-gpt2")
logger.info("Final model saved.")

# Evaluate the model
logger.info("Evaluating the model...")
eval_results = trainer.evaluate()
logger.info(f"Model evaluation completed. Perplexity: {eval_results.get('perplexity', 'N/A')}")
print(f"Perplexity: {eval_results.get('perplexity', 'N/A')}")
