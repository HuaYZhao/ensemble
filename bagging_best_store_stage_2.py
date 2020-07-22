# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 10:14
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import pickle
import json
import collections
import numpy as np
from functional import seq
from pprint import pprint
import itertools
from sklearn import preprocessing
from eval import main2
from multiprocessing import Pool as ProcessPool

models = [
    "albert_args_train_answer_results/1_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_answer_results/1_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_answer_results/1_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_answer_results/1_albert_xxlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_answer_results/2_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_answer_results/2_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_answer_results/2_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_answer_results/2_albert_xxlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_answer_results/3_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_answer_results/3_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_answer_results/3_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_answer_results/3_albert_xxlarge_v2_32_384_2e-05_2_0",
    "args_train_pv_results/1_electra_large_24_480_3e-05_2_0",
    "args_train_pv_results/1_electra_large_24_512_5e-05_2_0",
    "args_train_pv_results/1_electra_large_32_384_5e-05_2_0",
    "args_train_pv_results/1_electra_large_32_480_5e-05_2_0",
    "args_train_pv_results/1_electra_large_32_512_2e-05_2_0",
    "args_train_pv_results/1_electra_large_32_512_5e-05_2_0",
    "args_train_pv_results/1_electra_large_48_256_6e-05_2_0",
    "args_train_pv_results/2_electra_large_24_480_3e-05_2_0",
    "args_train_pv_results/2_electra_large_24_512_5e-05_2_0",
    "args_train_pv_results/2_electra_large_32_384_5e-05_2_0",
    "args_train_pv_results/2_electra_large_32_480_5e-05_2_0",
    "args_train_pv_results/2_electra_large_32_512_2e-05_2_0",
    "args_train_pv_results/2_electra_large_32_512_5e-05_2_0",
    "args_train_pv_results/2_electra_large_48_256_6e-05_2_0",
    "args_train_pv_results/3_electra_large_24_480_3e-05_2_0",
    "args_train_pv_results/3_electra_large_24_512_5e-05_2_0",
    "args_train_pv_results/3_electra_large_32_384_5e-05_2_0",
    "args_train_pv_results/3_electra_large_32_480_5e-05_2_0",
    "args_train_pv_results/3_electra_large_32_512_2e-05_2_0",
    "args_train_pv_results/3_electra_large_32_512_5e-05_2_0",
    "args_train_pv_results/3_electra_large_48_256_6e-05_2_0",
]

models = [models[x] for x in [3, 4, 7, 10, 17, 20, 22, 30]]

pprint(models)
all_odds = []
for dire in [os.path.join(d) for d in models]:
    if "albert" in dire:
        all_odds.append(json.load(
            open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))
    else:
        all_odds.append(json.load(
            open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))

qid_answers = collections.OrderedDict()
qid_questions = collections.OrderedDict()
for article in json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data']:
    for paragraph in article["paragraphs"]:
        for qa in paragraph['qas']:
            _qid = qa['id']
            qid_answers[_qid] = qa['answers']
            qid_questions[_qid] = qa['question']

base_preds = json.load(open("bagging_preds.json", 'r', encoding='utf-8'))
base_null_odds = json.load(open("bagging_odds.json", 'r', encoding='utf-8'))


def vote1():
    """

    """
    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in qid_answers:
        bagging_preds[qid] = base_preds[qid]
        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
    json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

    xargs = f"python eval.py dev-v2.0.json bagging_preds.json --na-prob-file bagging_odds.json"
    os.system(xargs)


vote1()
