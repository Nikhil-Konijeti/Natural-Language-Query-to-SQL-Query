import json
def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print("Loading data from %s"%SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print("Loading data from %s"%TABLE_PATH)
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab['id']] = tab

    for sql in sql_data:
        assert sql['table_id'] in table_data

    return sql_data, table_data


def load_dataset(dataset_id, use_small=False):
    if dataset_id == 0:
        sql_data, table_data = load_data('data/train_tok.jsonl',
                'data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
                'data/dev_tok.tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data/test_tok.jsonl',
                'data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data/train.db'
        DEV_DB = 'data/dev.db'
        TEST_DB = 'data/test.db'
    else:
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = load_dataset(1)

import matplotlib.pyplot as plt
import pandas as pd
import torch



# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'':0,
           'MAX':1,
          'MIN' :2,
          'COUNT':3,
          'SUM' :4,
          'AVG' : 5
          }
import pandas as pd
df_train_q = []
df_train_agg = []

df_val_q, df_val_agg = [], []

for i in range(len(sql_data)) :
  df_train_q.append(sql_data[i]['question'])
  df_train_agg.append(sql_data[i]['sql']['agg'])

for i in range(len(val_sql_data)) :
    df_val_q.append(val_sql_data[i]['question'])
    df_val_agg.append(val_sql_data[i]['sql']['agg'])

df_train = pd.DataFrame({'question':df_train_q, 'agg' : df_train_agg })
df_val = pd.DataFrame({'question':df_val_q, 'agg' : df_val_agg })

print(df_train['agg'], len(df_val))

from torch import nn
from transformers import BertModel
from sklearn.metrics import f1_score

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

from transformers import RobertaTokenizer, RobertaModel

class RoBertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 6)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer




class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          ### New layers:
          self.lstm = nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.linear = nn.Linear(256*2, 6)


    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(
               ids,
               attention_mask=mask)

          lstm_output, (h,c) = self.lstm(sequence_output)
          hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
          linear_output = self.linear(lstm_output[:,-1].view(-1,256*2))

          return linear_output

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        print(df['agg'])
        self.labels = [label for label in df['agg']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['question']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
from torch.optim import Adam
from tqdm import tqdm
def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)
    print(train[:0])

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            i = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

EPOCHS = 5
model1 = BertClassifier()
#model2 = CustomBERTModel()
LR = 1e-6

train(model1, df_train, df_val, LR, EPOCHS)
#train(model2, df_train, df_val, LR, EPOCHS)


import pandas as pd
df_train_q_sel = []
df_train_sel = []
df_val_q_sel, df_val_sel, = [], []

for i in range(len(sql_data)) :
  df_train_q_sel.append(sql_data[i]['question'])
  df_train_sel.append(int(sql_data[i]['sql']['sel']))

for i in range(len(val_sql_data)) :
    df_val_q_sel.append(val_sql_data[i]['question'])
    df_val_sel.append(int(val_sql_data[i]['sql']['sel']))

max_sel = max(max(df_train_sel), max(df_val_sel))

df_train_sel = pd.DataFrame({'question':df_train_q_sel, 'sel' : df_train_sel })
df_val_sel = pd.DataFrame({'question':df_val_q_sel, 'sel' : df_val_sel })

print(df_val_sel)


class DatasetSel(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [label for label in df['sel']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['question']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

from torch.optim import Adam
from tqdm import tqdm

class BertClassifierSel(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifierSel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 41)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer



train, val = DatasetSel(df_train_sel), DatasetSel(df_val_sel)
train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)


def trainSel(model, df_train_sel, df_val_sel, learning_rate, epochs):
    train, val = DatasetSel(df_train_sel), DatasetSel(df_val_sel)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)


    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            i = 0
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                output = model(input_id, mask)
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

EPOCHS = 5
LR = 1e-6

model3 = BertClassifierSel()
trainSel(model3,df_train_sel, df_val_sel, LR, EPOCHS )
