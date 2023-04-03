from transformers import RobertaForMaskedLM, DataCollatorForLanguageModeling, RobertaConfig, Trainer, TrainingArguments, RobertaTokenizer
from tokenizers import Tokenizer
from tokenizers.processors import RobertaProcessing
from datasets import Dataset
import tokenizers
import glob
import pickle

# load tokeniser, dataset
tokenizer = RobertaTokenizer("tokenisers/bpe/vocab.json", "tokenisers/bpe/merges.txt")
tokenizer.mask_token = "<mask>"
dataset: Dataset
with open("babylm_data/babylm_10M.pkl", "rb") as f:
    dataset = pickle.load(f)

# MLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# model
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512 + 2,
    num_attention_heads=8,
    num_hidden_layers=4,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    hidden_size=128
)
model = RobertaForMaskedLM(config)
print(f"Parameters: {model.num_parameters() / 1e6}M")

# training
training_args = TrainingArguments(
    output_dir="roberta_babylm_10M",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    save_steps=10_000,
    learning_rate=2e-5,
    weight_decay=0.01,
    optim="adamw_torch",
    gradient_checkpointing=True,
    eval_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()