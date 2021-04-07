import re

x_text = ['']
with open('data.txt', 'r') as datafile:
    lines = datafile.readlines()
    for line in lines:
        if(line != '\n'):
            line = re.sub('[\n]','',line)
            line = re.sub('["]','',line)
            x_text[-1] += line
        else:
            x_text.append('')

with open('y_1.txt', 'r') as datafile:
    y1_label = datafile.readlines()
    for i in range(len(y1_label)):
        y1_label[i] = re.sub('[\n]','',y1_label[i])
        y1_label[i] = int(y1_label[i])

from transformers import BertTokenizer, BertForSequenceClassification
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

from sklearn.model_selection import train_test_split
x_train_texts, x_test_texts, y1_train_label, y1_test_label = train_test_split(x_text, y1_label, test_size=.2)



tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
train_encodings = tokenizer(x_train_texts,padding=True, is_split_into_words=False,truncation=True)
test_encodings = tokenizer(x_test_texts,padding=True, is_split_into_words=False,truncation=True)
y1_train_labels = torch.tensor(y1_train_label)
y1_test_labels = torch.tensor(y1_test_label)

train_dataset = Dataset(train_encodings, y1_train_labels)
test_dataset = Dataset(test_encodings, y1_test_labels)



from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-7)

for epoch in range(20):
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
