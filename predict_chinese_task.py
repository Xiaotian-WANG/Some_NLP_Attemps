from pathlib import Path
import re

def read_wnut(file_path):
    file_path = Path(file_path)

    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

texts, tags = read_wnut('data/military_title_ner_data.txt')

from sklearn.model_selection import train_test_split
train_texts, val_texts, train_tags, val_tags = train_test_split(texts, tags, test_size=.2)

unique_tags = set(tag for doc in tags for tag in doc)
tag2id = {tag: id for id, tag in enumerate(unique_tags)}
id2tag = {id: tag for tag, id in tag2id.items()}

import numpy as np


import torch

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ============================

for i in range(len(train_tags)):
    for j in range(len(train_tags[i])):
        train_tags[i][j]=tag2id[train_tags[i][j]]

for i in range(len(val_tags)):
    for j in range(len(val_tags[i])):
        val_tags[i][j]=tag2id[val_tags[i][j]]


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(train_texts, is_split_into_words=True, padding=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, padding=True)

for i in range(train_encodings.data['input_ids'].__len__()):
    num_zeros = train_encodings.data['input_ids'][i].__len__()-train_tags[i].__len__()
    temp = np.hstack((np.array(train_tags[i]), (np.zeros(num_zeros, dtype=int)+tag2id['O'])))
    train_tags[i] = list(temp)

for i in range(val_encodings.data['input_ids'].__len__()):
    num_zeros = val_encodings.data['input_ids'][i].__len__()-val_tags[i].__len__()
    temp = np.hstack((np.array(val_tags[i]), (np.zeros(num_zeros, dtype=int)+tag2id['O'])))
    val_tags[i] = list(temp)



train_dataset = WNUTDataset(train_encodings, train_tags)
val_dataset = WNUTDataset(val_encodings, val_tags)



from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(unique_tags),return_dict=True)

from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()
    print(epoch, loss.tolist())

model.eval()

print("finished")

'''
import numpy as np
test_sample = val_dataset[19]
with torch.no_grad():
    output = model(input_ids=test_sample['input_ids'].unsqueeze(0).to(device),attention_mask=test_sample['attention_mask'].unsqueeze(0).to(device))

output = output[0].squeeze(0)

mylabel = list()

for i in range(output.size()[0]):
    mylabel.append(output[i].argmax().tolist())

thelabel = test_sample['labels'].tolist()

print(np.vstack((np.array(mylabel),np.array(thelabel))))
print(tokenizer.decode(test_sample['input_ids']))

'''

