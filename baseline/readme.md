Baseline Models

## Citation

> Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning.

## Bibtex

```
@article{xu2017sqlnet,
  title={SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning},
  author={Xu, Xiaojun and Liu, Chang and Song, Dawn},
  journal={arXiv preprint arXiv:1711.04436},
  year={2017}
}
```


The data is in ` data_sqlnet.tar.bz2`. Unzip the code by running
```bash
tar -xjvf  data_sqlnet.tar.bz2
```

```bash
pip install -r requirements.txt
```

Additionally install the bert-embedding and the transformers library

1) LSTM/GRU without BERT
## Train

python get_embeddings.py //for storing embeddings

python train_no_bert.py

The AggPredictor, WordEmbedding, SelPredictor, SQLNetCondPredictor, utils.py and train.py have been modified.

2) LSTM with BERT / ROBERTA
This link was referred for the code:
https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

python train_no_bert.py

3)  evauluate BERT model :
python query_match.py
