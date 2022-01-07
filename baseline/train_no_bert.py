
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

def best_model_name(args, for_load=False):
    new_data = 'new' if args.dataset > 0 else 'old'
    mode =  'sqlnet'
    if for_load:
        use_emb = use_rl = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''
        use_rl =  ''
    use_ca = '_ca' if args.ca else ''

    agg_model_name = 'saved_model/%s_%s%s%s.agg_model'%(new_data,
            mode, use_emb, use_ca)
    sel_model_name = 'saved_model/%s_%s%s%s.sel_model'%(new_data,
            mode, use_emb, use_ca)
    cond_model_name = 'saved_model/%s_%s%s%s.cond_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)

    if not for_load and args.train_emb:
        agg_embed_name = 'saved_model/%s_%s%s%s.agg_embed'%(new_data,
                mode, use_emb, use_ca)
        sel_embed_name = 'saved_model/%s_%s%s%s.sel_embed'%(new_data,
                mode, use_emb, use_ca)
        cond_embed_name = 'saved_model/%s_%s%s%s.cond_embed'%(new_data,
                mode, use_emb, use_ca)

        return agg_model_name, sel_model_name, cond_model_name,\
                agg_embed_name, sel_embed_name, cond_embed_name
    else:
        return agg_model_name, sel_model_name, cond_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)

def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)

def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data,
                pred_queries, query_gt, pred_entry)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def epoch_reinforce_train(model, optimizer, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.train()
    perm = np.random.permutation(len(sql_data))
    cum_reward = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data =\
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, (True, True, True),
                reinforce=True, gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq,
                raw_col_seq, (True, True, True), reinforce=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        rewards = []
        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(1)

        cum_reward += (sum(rewards))
        optimizer.zero_grad()
        model.reinforce_backward(score, rewards)
        optimizer.step()

        st = ed

    return cum_reward / len(sql_data)


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print(('Loading word embedding from %s'%file_name))
        ret = {}
        with open(file_name,encoding="utf-8") as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array([float(x) for x in info[1:]])
        return ret
    else:
        print ('Load used word embedding')
        with open('bert/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('bert/usedwordemb.npy','rb') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val


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

sql_data, table_data, val_sql_data, val_table_data,\
        test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB = \
        load_dataset(0, use_small=USE_SMALL)
word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
            load_used=True, use_small=USE_SMALL)


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import traceback

class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
            our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print("Using trainable embedding")
            self.w2i, word_emb_val = word_emb
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                    torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print("Using fixed embedding")


    def gen_x_batch(self, q, col):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        try:
            for i, (one_q, one_col) in enumerate(zip(q, col)):
                if self.trainable:
                    q_val = [self.w2i.get(x, 0) for x in one_q]
                else:
                    q_val = [self.word_emb[1][self.word_emb[0][x]] if x in self.word_emb[0]  else np.zeros(self.N_word, dtype=np.float32) for x in one_q]
                if self.our_model:
                    if self.trainable:
                        val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                    else:
                        val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                    val_len[i] = 1 + len(q_val) + 1
                else:
                    one_col_all = [x for toks in one_col for x in toks+[',']]
                    if self.trainable:
                        col_val = [self.w2i.get(x, 0) for x in one_col_all]
                        val_embs.append( [0 for _ in self.SQL_TOK] + col_val + [0] + q_val+ [0])
                    else:
                        col_val =[self.word_emb[1][self.word_emb[0][x]] if x in self.word_emb[0]  else np.zeros(self.N_word, dtype=np.float32) for x in  one_col_all]
                        val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + col_val + [np.zeros(self.N_word, dtype=np.float32)] + q_val+ [np.zeros(self.N_word, dtype=np.float32)])
                    val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        except :
            print("error")
            traceback.print_exc()

        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in one_str]
            else:
                val = [self.word_emb[1][self.word_emb[0][x]] if x in self.word_emb[0]  else np.zeros(self.N_word, dtype=np.float32) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(list(range(len(inp_len))),
        key=lambda k:inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
            sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    #Encode the columns.
    #The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(
            len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len

def att_flow_layer(c, q,softmax,att_weight_cq,att_weight_c,att_weight_q):
        """
        :param c: (batch, c_len, hidden_size * 2)
        :param q: (batch, q_len, hidden_size * 2)
        :return: (batch, c_len, q_len)
        """
        c_len = c.size(1)
        q_len = q.size(1)

        # (batch, c_len, q_len, hidden_size * 2)
        #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
        # (batch, c_len, q_len, hidden_size * 2)
        #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
        # (batch, c_len, q_len, hidden_size * 2)
        #cq_tiled = c_tiled * q_tiled
        #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

        cq = []
        for i in range(q_len):

            #(batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            #(batch, c_len, 1)
            ci = att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = att_weight_c(c).expand(-1, -1, q_len) + \
            att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)

        x=c + c2q_att+ c * c2q_att + c * q2c_att
        y=softmax(x)
        return y

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.GRU(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        
        if use_ca:
            print("Using column attention on selection predicting")
            self.sel_att = nn.Linear(N_h, N_h)
        else:
            print("Not using column attention on selection predicting")
            self.sel_att = nn.Linear(N_h, 1)
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()

        # BIDAF
        self.att_weight_c = nn.Linear(N_h, 1)
        self.att_weight_q = nn.Linear(N_h, 1)
        self.att_weight_cq = nn.Linear(N_h, 1)

    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)

        if self.use_ca:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)


            h_enc = att_flow_layer(h_enc, e_col,self.softmax,self.att_weight_cq,self.att_weight_c,self.att_weight_q)


            att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            att = self.softmax(att_val.view((-1, max_x_len))).view(
                    B, -1, max_x_len)
            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        else:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = self.sel_att(h_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val)
            K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            K_sel_expand=K_sel.unsqueeze(1)

        sel_score = self.sel_out( self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col) ).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca):
        super(AggPredictor, self).__init__()
        use_ca = False
        self.use_ca = use_ca
        self.dropout = nn.Dropout(0.5)
        self.linear_bert = nn.Linear(768, 5)


        self.agg_lstm = nn.GRU(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,dropout=0.3, bidirectional=True)
        if use_ca:
            print("Using column attention on aggregator predicting")
            self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                    hidden_size=(int)(N_h/2), num_layers=N_depth,
                    batch_first=True, dropout=0.3, bidirectional=True)
            self.agg_att = nn.Linear(N_h, N_h)
        else:
            print("Not using column attention on aggregator predicting")
            self.agg_att = nn.Linear(N_h, 1)
        self.agg_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 6))
        self.softmax = nn.Softmax()

        # BIDAF
        self.att_weight_c = nn.Linear(N_h, 1)
        self.att_weight_q = nn.Linear(N_h, 1)
        self.att_weight_cq = nn.Linear(N_h, 1)

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
            col_len=None, col_num=None, gt_sel=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)
        if self.use_ca:
            h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
            e_col, _ = col_name_encode(col_inp_var, col_name_len, 
                    col_len, self.agg_col_name_enc)

            h_enc = att_flow_layer(h_enc, e_col,self.softmax,self.att_weight_cq,self.att_weight_c,self.att_weight_q)

            chosen_sel_idx = torch.LongTensor(gt_sel)
            aux_range = torch.LongTensor(list(range(len(gt_sel))))
            if x_emb_var.is_cuda:
                chosen_sel_idx = chosen_sel_idx.cuda()
                aux_range = aux_range.cuda()
            chosen_e_col = e_col[aux_range, chosen_sel_idx]
            att_val = torch.bmm(self.agg_att(h_enc), 
                    chosen_e_col.unsqueeze(2)).squeeze()
        else:
            sort_perm = np.array(sorted(list(range(len(x_len))),
            key=lambda k:x_len[k], reverse=True))
            sort_inp_len = x_len[sort_perm]
            sort_perm_inv = np.argsort(sort_perm)
            if x_emb_var.is_cuda:
                sort_perm = torch.LongTensor(sort_perm).cuda()
                sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

            lstm_inp = nn.utils.rnn.pack_padded_sequence(x_emb_var[sort_perm],
                    sort_inp_len, batch_first=True)
            sort_ret_s, sort_ret_h = self.agg_lstm(lstm_inp, None)
            ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
            ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
            h_enc, _ = ret_s, ret_h   

            att_val = self.agg_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)

        K_agg = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        agg_score = self.agg_out(K_agg)
        return agg_score


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class SQLNetCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca, gpu):
        super(SQLNetCondPredictor, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu
        self.use_ca = use_ca

        self.cond_num_lstm = nn.GRU(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_num_att = nn.Linear(N_h, 1)
        self.cond_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 5))
        self.cond_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_num_col_att = nn.Linear(N_h, 1)
        self.cond_num_col2hid1 = nn.Linear(N_h, 2*N_h)
        self.cond_num_col2hid2 = nn.Linear(N_h, 2*N_h)

        self.cond_col_lstm = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            print("Using column attention on where predicting")
            self.cond_col_att = nn.Linear(N_h, N_h)
        else:
            print("Not using column attention on where predicting")
            self.cond_col_att = nn.Linear(N_h, 1)
        self.cond_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.cond_op_lstm = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            self.cond_op_att = nn.Linear(N_h, N_h)
        else:
            self.cond_op_att = nn.Linear(N_h, 1)
        self.cond_op_out_K = nn.Linear(N_h, N_h)
        self.cond_op_name_enc = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_op_out_col = nn.Linear(N_h, N_h)
        self.cond_op_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 3))

        self.cond_str_lstm = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)
        self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=(int)(N_h/2),
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.cond_str_out_g = nn.Linear(N_h, N_h)
        self.cond_str_out_h = nn.Linear(N_h, N_h)
        self.cond_str_out_col = nn.Linear(N_h, N_h)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()

        # BIDAF
        self.att_weight_c = nn.Linear(N_h, 1)
        self.att_weight_q = nn.Linear(N_h, 1)
        self.att_weight_cq = nn.Linear(N_h, 1)

    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)
        max_len = max([max([len(tok) for tok in tok_seq]+[0]) for 
            tok_seq in split_tok_seq]) - 1 # The max seq len in the batch.
        if max_len < 1:
            max_len = 1
        ret_array = np.zeros((
            B, 4, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 4))
        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1
            if idx < 3:
                ret_array[b, idx+1:, 0, 1] = 1
                ret_len[b, idx+1:] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp)

        return ret_inp_var, ret_len #[B, IDX, max_len, max_tok_num]


    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
            col_len, col_num, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)
        if reinforce:
            raise NotImplementedError('Our model doesn\'t have RL')

        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_num_name_enc)
        num_col_att_val = self.cond_num_col_att(e_num_col).squeeze()
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        cond_num_h1 = self.cond_num_col2hid1(K_num_col).view(
                B, 4, (int)(self.N_h/2)).transpose(0, 1).contiguous()
        cond_num_h2 = self.cond_num_col2hid2(K_num_col).view(
                B, 4, (int)(self.N_h/2)).transpose(0, 1).contiguous()

        # h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len,
        #         hidden=(cond_num_h1, cond_num_h2))
        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len)

        num_att_val = self.cond_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)
        cond_num_score = self.cond_num_out(K_cond_num)

        #Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len,
                self.cond_col_name_enc)

        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len)

        h_col_enc = att_flow_layer(h_col_enc, e_cond_col, self.softmax, self.att_weight_cq, self.att_weight_c, self.att_weight_q)

        if self.use_ca:
            col_att_val = torch.bmm(e_cond_col,
                    self.cond_col_att(h_col_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, :, num:] = -100
            col_att = self.softmax(col_att_val.view(
                (-1, max_x_len))).view(B, -1, max_x_len)
            K_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)
        else:
            col_att_val = self.cond_col_att(h_col_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, num:] = -100
            col_att = self.softmax(col_att_val)
            K_cond_col = (h_col_enc *
                    col_att_val.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                self.cond_col_out_col(e_cond_col)).squeeze()
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        #Predict the operator of conditions
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(), axis=1)
            col_scores = cond_col_score.data.cpu().numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]])
                    for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [ [x[0] for x in one_gt_cond] for 
                    one_gt_cond in gt_cond]

        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_op_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x] 
                for x in chosen_col_gt[b]] + [e_cond_col[b, 0]] *
                (4 - len(chosen_col_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len)
        if self.use_ca:
            op_att_val = torch.matmul(self.cond_op_att(h_op_enc).unsqueeze(1),
                    col_emb.unsqueeze(3)).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, :, num:] = -100
            op_att = self.softmax(op_att_val.view(B*4, -1)).view(B, 4, -1)
            K_cond_op = (h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)
        else:
            op_att_val = self.cond_op_att(h_op_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, num:] = -100
            op_att = self.softmax(op_att_val)
            K_cond_op = (h_op_enc * op_att.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) +
                self.cond_op_out_col(col_emb)).squeeze()

        #Predict the string of conditions
        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.cond_str_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x]
                for x in chosen_col_gt[b]] +
                [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
            g_str_s_flat, _ = self.cond_str_decoder(
                    gt_tok_seq.view(B*4, -1, self.max_tok_num))
            g_str_s = g_str_s_flat.contiguous().view(B, 4, -1, self.N_h)

            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            g_ext = g_str_s.unsqueeze(3)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)

            cond_str_score = self.cond_str_out(
                    self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                    self.cond_str_out_col(col_ext)).squeeze()
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100
        else:
            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)
            scores = []

            t = 0
            init_inp = np.zeros((B*4, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:,0,0] = 1  #Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = None
            while t < 50:
                if cur_h:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                else:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                g_str_s = g_str_s_flat.view(B, 4, 1, self.N_h)
                g_ext = g_str_s.unsqueeze(3)

                cur_cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                        + self.cond_str_out_col(col_ext)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_str_score[b, :, num:] = -100
                scores.append(cur_cond_str_score)

                _, ans_tok_var = cur_cond_str_score.view(B*4, max_x_len).max(1)
                ans_tok = ans_tok_var.data.cpu()
                data = torch.zeros(B*4, self.max_tok_num).scatter_(
                        1, ans_tok.unsqueeze(1), 1)
                if self.gpu:  #To one-hot
                    cur_inp = Variable(data.cuda())
                else:
                    cur_inp = Variable(data)
                cur_inp = cur_inp.unsqueeze(1)

                t += 1

            cond_str_score = torch.stack(scores, 2)
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100  #[B, IDX, T, TOK_NUM]

        cond_score = (cond_num_score,
                cond_col_score, cond_op_score, cond_str_score)

        return cond_score


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
            gpu=False, use_ca=True, trainable_emb=False):
        super(SQLNet, self).__init__()
        self.use_ca = use_ca
        self.trainable_emb = trainable_emb

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
                'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        #Word embedding
        if trainable_emb:
            self.agg_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
            self.sel_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
            self.cond_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
        else:
            self.embed_layer = WordEmbedding(word_emb, N_word, gpu,
                    self.SQL_TOK, our_model=True, trainable=trainable_emb)
        
        #Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=use_ca)

        #Predict selected column
        self.sel_pred = SelPredictor(N_word, N_h, N_depth,
                self.max_tok_num, use_ca=use_ca)

        #Predict number of cond
        self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth,
                self.max_col_num, self.max_tok_num, use_ca, gpu)



        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()


    def generate_gt_where_seq(self, q, col, query):
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            cur_values = []
            st = cur_query.index('WHERE')+1 if \
                    'WHERE' in cur_query else len(cur_query)
            all_toks = ['<BEG>'] + cur_q + ['<END>']
            while st < len(cur_query):
                ed = len(cur_query) if 'AND' not in cur_query[st:]\
                        else cur_query[st:].index('AND') + st
                if 'EQL' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('EQL') + st
                elif 'GT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('GT') + st
                elif 'LT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('LT') + st
                else:
                    raise RuntimeError("No operator in it!")
                this_str = ['<BEG>'] + cur_query[op+1:ed] + ['<END>']
                cur_seq = [all_toks.index(s) if s in all_toks \
                        else 0 for s in this_str]
                cur_values.append(cur_seq)
                st = ed+1
            ret_seq.append(cur_values)
        return ret_seq


    def forward(self, q, col, col_num, pred_entry,
            gt_where = None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None

        #Predict aggregator
        if self.trainable_emb:
            if pred_agg:
                x_emb_var, x_len = self.agg_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.agg_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel=gt_sel)

            if pred_sel:
                x_emb_var, x_len = self.sel_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.sel_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num)

            if pred_cond:
                x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
                col_inp_var, col_name_len, col_len = \
                        self.cond_embed_layer.gen_col_batch(col)
                max_x_len = max(x_len)
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num,
                        gt_where, gt_cond, reinforce=reinforce)
        else:
            x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
            col_inp_var, col_name_len, col_len = \
                    self.embed_layer.gen_col_batch(col)
            max_x_len = max(x_len)
            if pred_agg:
                agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num, gt_sel=gt_sel)   #[64,6]

            if pred_sel:
                sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num)

            if pred_cond:
                cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var,
                        col_name_len, col_len, col_num,
                        gt_where, gt_cond, reinforce=reinforce)

        return (agg_score, sel_score, cond_score)

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        loss = 0
        if pred_agg:
            agg_truth = [x[0] for x in truth_num]
            data = torch.from_numpy(np.array(agg_truth))
            if self.gpu:
                agg_truth_var = Variable(data.cuda())
            else:
                agg_truth_var = Variable(data)

            loss += self.CE(agg_score, agg_truth_var.long())

        if pred_sel:
            sel_truth = [x[1] for x in truth_num]
            data = torch.from_numpy(np.array(sel_truth))
            if self.gpu:
                sel_truth_var = Variable(data.cuda())
            else:
                sel_truth_var = Variable(data)

            loss += self.CE(sel_score, sel_truth_var.long())

        if pred_cond:
            B = len(truth_num)
            cond_num_score, cond_col_score,\
                    cond_op_score, cond_str_score = cond_score
            #Evaluate the number of conditions
            cond_num_truth = [x[2] for x in truth_num]
            data = torch.from_numpy(np.array(cond_num_truth))
            if self.gpu:
                cond_num_truth_var = Variable(data.cuda())
            else:
                cond_num_truth_var = Variable(data)
            loss += self.CE(cond_num_score, cond_num_truth_var.long())

            #Evaluate the columns of conditions
            T = len(cond_col_score[0])
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):
                if len(truth_num[b][3]) > 0:
                    truth_prob[b][list(truth_num[b][3])] = 1
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                cond_col_truth_var = Variable(data.cuda())
            else:
                cond_col_truth_var = Variable(data)

            sigm = nn.Sigmoid()
            cond_col_prob = sigm(cond_col_score)
            bce_loss = -torch.mean( 3*(cond_col_truth_var * \
                    torch.log(cond_col_prob+1e-10)) + \
                    (1-cond_col_truth_var) * torch.log(1-cond_col_prob+1e-10) )
            loss += bce_loss

            #Evaluate the operator of conditions
            for b in range(len(truth_num)):
                if len(truth_num[b][4]) == 0:
                    continue
                data = torch.from_numpy(np.array(truth_num[b][4]))
                if self.gpu:
                    cond_op_truth_var = Variable(data.cuda())
                else:
                    cond_op_truth_var = Variable(data)
                cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
                loss += (self.CE(cond_op_pred, cond_op_truth_var.long()) \
                        / len(truth_num))

            #Evaluate the strings of conditions
            for b in range(len(gt_where)):
                for idx in range(len(gt_where[b])):
                    cond_str_truth = gt_where[b][idx]
                    if len(cond_str_truth) == 1:
                        continue
                    data = torch.from_numpy(np.array(cond_str_truth[1:]))
                    if self.gpu:
                        cond_str_truth_var = Variable(data.cuda())
                    else:
                        cond_str_truth_var = Variable(data)
                    str_end = len(cond_str_truth)-1
                    cond_str_pred = cond_str_score[b, idx, :str_end]
                    loss += (self.CE(cond_str_pred, cond_str_truth_var.long()) \
                            / (len(gt_where) * len(gt_where[b])))

        return loss

    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        def pretty_print(vis_data):
            print('question:', vis_data[0])
            print('headers: (%s)'%(' || '.join(vis_data[1])))
            print('query:', vis_data[2])

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                    self.COND_OPS[cond[1]] + ' ' + str(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(x[0] for x in cond_pred) != \
                        set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(
                            x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(
                            x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and str(cond_gt[gt_idx][2]).lower() != \
                            str(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err


    def gen_query(self, score, q, col, raw_q, raw_col,
            pred_entry, reinforce=False, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(',
                    '-RRB-':')',
                    '-LSB-':'[',
                    '-RSB-':']',
                    '``':'"',
                    '\'\'':'"',
                    '--':'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        ret_queries = []
        if pred_agg:
            B = len(agg_score)
        elif pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0])
        for b in range(B):
            cur_query = {}
            if pred_agg:
                cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            if pred_cond:
                cur_query['conds'] = []
                cond_num_score,cond_col_score,cond_op_score,cond_str_score =\
                        [x.data.cpu().numpy() for x in cond_score]
                cond_num = np.argmax(cond_num_score[b])
                all_toks = ['<BEG>'] + q[b] + ['<END>']
                max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
                for idx in range(cond_num):
                    cur_cond = []
                    cur_cond.append(max_idxes[idx])
                    cur_cond.append(np.argmax(cond_op_score[b][idx]))
                    cur_cond_str_toks = []
                    for str_score in cond_str_score[b][idx]:
                        str_tok = np.argmax(str_score[:len(all_toks)])
                        str_val = all_toks[str_tok]
                        if str_val == '<END>':
                            break
                        cur_cond_str_toks.append(str_val)
                    cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))
                    cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries


from sklearn.metrics import f1_score

def get_f1( vis_info, pred_queries, gt_queries, pred_entry):

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        agg_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        pred_val = []
        gt_val = []
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            a = np.zeros(5)
            a[pred_qry['agg']] = 1
            b = np.zeros(5)
            b[gt_qry['agg']] = 1
            pred_val.append(a)
            gt_val.append(b)
        score = f1_score(gt_val, pred_val, average=None, zero_division=1 )

        return score

def epoch_f1(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    curr_f1 = []
    tries = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        curr_f1.append(get_f1(raw_data,
                pred_queries, query_gt, pred_entry))
        tries = tries + 1
        st = ed
    avg_f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(len(curr_f1)) :
        for j in range(5) :
          avg_f1[j] = avg_f1[j] + curr_f1[i][j]
    
    for j in range(5) :
          avg_f1[j] = avg_f1[j] / float(tries)
    
    print("val f1", sum(avg_f1)/5)
    return sum(avg_f1)/5

N_word=768
B_word=6
toy = True
if toy:
    USE_SMALL=True
    GPU=True
    BATCH_SIZE=15
else:
    USE_SMALL=False
    GPU=True
    BATCH_SIZE=64
TRAIN_ENTRY=(True, True, True)  # (AGG, SEL, COND)
TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
learning_rate =  1e-3
import warnings
warnings.filterwarnings('ignore')

sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    0, use_small=USE_SMALL)


model = SQLNet(word_emb, N_word=N_word, use_ca=True,
                gpu=GPU, trainable_emb = False)
optimizer = torch.optim.Adam(model.parameters(),
            lr=learning_rate, weight_decay = 0)

new_data = 'new'
mode = 'sqlnet'
use_emb = ''
use_ca = ''
use_rl = ''
agg_model_name = 'saved_model/%s_%s%s%s.agg_model'%(new_data,
            mode, use_emb, use_ca)
sel_model_name = 'saved_model/%s_%s%s%s.sel_model'%(new_data,
            mode, use_emb, use_ca)
cond_model_name = 'saved_model/%s_%s%s%s.cond_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)


agg_m, sel_m, cond_m = agg_model_name, sel_model_name, cond_model_name

if True:
        init_acc = epoch_acc(model, BATCH_SIZE,
                val_sql_data, val_table_data, TRAIN_ENTRY)
        best_agg_acc = init_acc[1][0]
        best_agg_idx = 0
        best_sel_acc = init_acc[1][1]
        best_sel_idx = 0
        best_cond_acc = init_acc[1][2]
        best_cond_idx = 0
        print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s'%\
                init_acc)
        if TRAIN_AGG:
            torch.save(model.agg_pred.state_dict(), agg_m)
        if TRAIN_SEL:
            torch.save(model.sel_pred.state_dict(), sel_m)
        if TRAIN_COND:
            torch.save(model.cond_pred.state_dict(), cond_m)
        for i in range(5):
            print('Epoch %d @ %s'%(i+1, datetime.datetime.now()))
            print(' Loss = %s'%epoch_train(
                    model, optimizer, BATCH_SIZE, 
                    sql_data, table_data, TRAIN_ENTRY))
            print(' Train acc_qm: %s\n   breakdown result: %s'%epoch_acc(
                    model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY))
            val_acc = epoch_acc(model,
                    BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
            
            epoch_f1(model,
                    BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
              
            print(' Dev acc_qm: %s\n   breakdown result: %s'%val_acc)


