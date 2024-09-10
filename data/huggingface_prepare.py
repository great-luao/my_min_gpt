from transformers import AutoTokenizer
import pandas as pd
import os
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), 'tokenizer'))
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'reviews.csv'))
    
# add propmt before each review base on the sentiment
def add_prompt(row):
    if row["sentiment"] == "1":
        return "好评：" + row["review"]
    else:
        return "差评：" + row["review"]

df["review"] = df.apply(add_prompt, axis=1)

def tokenize_text(sequence):
    """Tokenize input sequence."""
    return tokenizer(sequence, padding=True, truncation=True, max_length=256)

# Tokenize all of the sentences and map the tokens to their word IDs
tok = df['review'].map(tokenize_text)
tok_df = pd.DataFrame(list(tok))

# store the tokenized data
tok_df.to_csv("tok_reviews.csv", index=False)

    