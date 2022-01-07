
import json
import torch
import numpy as np
import datetime

LOCAL_TEST=True


if LOCAL_TEST:
    N_word=768
    B_word=6
    USE_SMALL=True
else:
    N_word=300
    B_word=42
    USE_SMALL=False


import json
import torch
import numpy as np
import datetime

import argparse

import json
#from .lib.dbengine import DBEngine
import re
import numpy as np
#from nltk.tokenize import StanfordTokenizer

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    a = 0 
    b = 0 
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
        print("Loading from original dataset")
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
        print("Loading from re-split dataset")
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

sql_data, table_data, val_sql_data, val_table_data,\
        test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = \
        load_dataset(0, use_small=USE_SMALL)


import json
import torch
import numpy as np
import datetime

import mxnet as mx
from bert_embedding import BertEmbedding
bert = BertEmbedding()

word_to_idx = {'<UNK>':0, '<BEG>':1, '<END>':2}
word_num = 3
embs = [np.zeros(N_word,dtype=np.float32) for _ in range(word_num)]

def check_and_add(tok):
    #Check if the tok is in the vocab. If not, add it.
    global word_num
    if tok not in word_to_idx:
        #print("token=\"",tok,[tok[1:] + " "])

        word_to_idx[tok] = word_num
        word_num += 1
        embs.append(bert([tok + " "],'sum')[0][1][0])

for sql in sql_data:
    for tok in sql['question_tok']:
        check_and_add(tok)
for tab in list(table_data.values()):
    for col in tab['header_tok']:
        for tok in col:
            check_and_add(tok)
for sql in val_sql_data:
    for tok in sql['question_tok']:
        check_and_add(tok)
for tab in list(val_table_data.values()):
    for col in tab['header_tok']:
        for tok in col:
            check_and_add(tok)
for sql in test_sql_data:
    for tok in sql['question_tok']:
        check_and_add(tok)
for tab in list(test_table_data.values()):
    for col in tab['header_tok']:
        for tok in col:
            check_and_add(tok)

print("Length of used word vocab: %s"%len(word_to_idx))
emb_array = np.stack(embs, axis=0)
with open('bert/word2idx.json', 'w') as outf:
    json.dump(word_to_idx, outf)
np.save(open('bert/usedwordemb.npy', 'wb'), emb_array)

