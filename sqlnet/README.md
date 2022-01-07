# SQLNET
The code is based on the SQLNet model with BERT word embeddings. The code uses Python 3.6 and pytorch 1.3.1. 

## Citation

> Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning.

Original SQLNET code : 
https://github.com/xiaojunxu/SQLNet

## Bibtex

```
@article{xu2017sqlnet,
  title={SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning},
  author={Xu, Xiaojun and Liu, Chang and Song, Dawn},
  journal={arXiv preprint arXiv:1711.04436},
  year={2017}
}
```

## Installation
The data is in `data.tar.bz2`. Unzip the code by running
```bash
tar -xjvf data.tar.bz2
```
```bash
pip install -r requirements.txt
```

bert-embedding and mxnet-cu92  should be installed too. 

## Extract the bert embedding for training.
Run the following command to process the pretrained glove embedding for training the word embedding:
```bash
python extract_vocab.py
```

## Train
The training script is `train.py`. To see the detailed parameters for running:
```bash
python train.py -h
```

Some typical usage are listed as below:

Train a SQLNet model with column attention and BERT embeddings:
```bash
!python train.py --ca --use_bert
```





