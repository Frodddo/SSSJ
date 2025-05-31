from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "SUTD has a 1:1 student-faculty ratio."
print(tokenizer.tokenize(text))