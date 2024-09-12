import os
import tiktoken
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

split = 0.9
data_file = "reviews.csv"
token_type = 'small' # 'small' or 'gpt2'

# download the tiny shakespeare dataset
data = pd.read_csv(data_file)
# add prompt to the review
# def add_prompt(row):
#     if row["sentiment"] == 1:
#         return "好评：" + row["review"]
#     else:
#         return "差评：" + row["review"]

# data['review'] = data.apply(add_prompt, axis=1)
print(data.head())
# split the data
train_data, val_data = train_test_split(data, train_size=split, shuffle=True)
# rearrange the train_data order base on the sentiment row
train_data = train_data.sort_values(by='sentiment')
# combine all review part into a txt format
train_data = "\n".join(train_data['review'].values)
val_data = "\n".join(val_data['review'].values)
print(train_data[:100])

# encode with tiktoken gpt2 bpe
if token_type == 'gpt':
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    stoi = enc._special_tokens
    itos = {i: enc.decode([i]) for i in range(vocab_size)}
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
elif token_type == 'small':
    chars = sorted(list(set(train_data+val_data)))
    # print(chars)
    # chars += ['<eos>']
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)

print(f"vocab size: {vocab_size:,}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# output a meta.pickle file, which stores the vocab size of the tokenizer
meta = {'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}
with open("meta.pkl", 'wb') as f:
    pickle.dump(meta, f)


# train has 116,349 tokens
# val has 13,175 tokens