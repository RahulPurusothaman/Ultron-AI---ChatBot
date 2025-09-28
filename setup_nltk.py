import nltk
from nltk.tokenize import TreebankWordTokenizer

# Safe tokenizer that doesn't look for punkt_tab
tokenizer = TreebankWordTokenizer()

text = "Hello! How are you?"
tokens = tokenizer.tokenize(text)
print(tokens)
