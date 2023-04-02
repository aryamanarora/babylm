from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import RobertaProcessing
from datasets import Dataset

import pickle
import glob

# tokeniser and trainer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    min_frequency=2
)

# get files
files = list(glob.glob("babylm_data/babylm_10M/*.train"))

# init tokenizer
tokenizer.train(files, trainer)

# post processing
tokenizer.post_processor = RobertaProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

# save
tokenizer.model.save("tokenisers/bpe")

# load dataset
lines = []
for file in glob.glob("babylm_data/babylm_10M/*.train"):
    with open(file, "r") as f:
        lines.extend(list(map(lambda x: {'text': x}, f.readlines())))

def encode(example):
    e = tokenizer.encode(example["text"])
    return {'text': example, 'input_ids': e.ids, 'token_type_ids': e.type_ids, 'attention_mask': e.attention_mask}

dataset = Dataset.from_list(lines).map(encode)
with open("babylm_data/babylm_10M.pkl", "wb") as f:
    pickle.dump(dataset, f)