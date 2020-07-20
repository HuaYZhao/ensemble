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

# vote1 models em: 89.455 f1: 91.662   ave: 90.558
# models = [
#     "atrlp_results/3",
#     "atrlp_results/2",
#     "atrlp_results/4",
#     # "atrlp_results/7",
#     "atrlp_results/8",
#     # "lr_epoch_results/1e-05_2_2",
#     # "lr_epoch_results/1e-05_3_1",
#     "lr_epoch_results/2e-05_2_3",
#     # "lr_epoch_results/2e-05_3_1",
#     # "lr_epoch_results/3.0000000000000004e-05_2_3",
#     # "lr_epoch_results/3.0000000000000004e-05_3_2",
#     # "lr_epoch_results/4e-05_2_2",
#     # "lr_epoch_results/4e-05_3_3",
#     # "lr_epoch_results/5e-05_2_2",
#     # "lr_epoch_results/5e-05_3_1",
#     "lr_epoch_results/6e-05_2_1",
#     # "lr_epoch_results/6e-05_3_1",
# ]

# vote1 models em: 89.497  f1: 91.737   ave: 90.617
# models = [
#     # "args_train_results/1_electra_large_24_512_5e-05_2_2",
#     # "args_train_results/1_electra_large_32_384_5e-05_2_1",
#     "args_train_results/1_electra_large_32_480_5e-05_2_1",
#     # "args_train_results/1_electra_large_32_512_5e-05_2_2",
#     # "args_train_results/2_electra_large_24_512_5e-05_2_1",
#     "args_train_results/2_electra_large_32_384_5e-05_2_2",
#     "args_train_results/2_electra_large_32_480_5e-05_2_2",
#     # "args_train_results/2_electra_large_32_512_5e-05_2_1",
#     # "atrlp_results/1",
#     "atrlp_results/2",
#     "atrlp_results/3",
#     # "atrlp_results/4",
#     # "atrlp_results/5",
#     # "atrlp_results/6",
#     "atrlp_results/7",
#     "atrlp_results/8",
#     "atrlp_results/9",
#     # "atrlp_results/10",
#     # "lr_epoch_results/1e-05_2_2",
#     # "lr_epoch_results/1e-05_3_1",
#     # "lr_epoch_results/2e-05_2_3",
#     # "lr_epoch_results/2e-05_3_1",
#     # "lr_epoch_results/3.0000000000000004e-05_2_3",
#     # "lr_epoch_results/3.0000000000000004e-05_3_2",
#     # "lr_epoch_results/4e-05_2_2",
#     # "lr_epoch_results/4e-05_3_3",
#     # "lr_epoch_results/5e-05_2_2",
#     # "lr_epoch_results/5e-05_3_1",
#     "lr_epoch_results/6e-05_2_1",
#     "lr_epoch_results/6e-05_3_1",
# ]

# vote1 models em: 89.825 f1: 92.059  ave: 90.942
# models = [
#     "args_train_results/1_electra_large_24_512_5e-05_2_2",
#     "args_train_results/1_electra_large_32_384_5e-05_2_1",
#     "args_train_results/1_electra_large_32_480_5e-05_2_1",
#     "args_train_results/1_electra_large_32_512_5e-05_2_2",
#     "args_train_results/2_electra_large_24_512_5e-05_2_1",
#     "args_train_results/2_electra_large_32_384_5e-05_2_2",
#     "args_train_results/2_electra_large_32_480_5e-05_2_2",
#     "args_train_results/2_electra_large_32_512_5e-05_2_1",
#     "atrlp_results/1",
#     "atrlp_results/2",
#     "atrlp_results/3",
#     "atrlp_results/4",
#     "atrlp_results/5",
#     "atrlp_results/6",
#     "atrlp_results/7",
#     "atrlp_results/8",
#     "atrlp_results/9",
#     "atrlp_results/10",
#     "lr_epoch_results/1e-05_2_2",
#     "lr_epoch_results/1e-05_3_1",
#     "lr_epoch_results/2e-05_2_3",
#     "lr_epoch_results/2e-05_3_1",
#     "lr_epoch_results/3.0000000000000004e-05_2_3",
#     "lr_epoch_results/3.0000000000000004e-05_3_2",
#     "lr_epoch_results/4e-05_2_2",
#     "lr_epoch_results/4e-05_3_3",
#     "lr_epoch_results/5e-05_2_2",
#     "lr_epoch_results/5e-05_3_1",
#     "lr_epoch_results/6e-05_2_1",
#     "lr_epoch_results/6e-05_3_1",
#     "albert_args_train_results/1_albert_base_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_base_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_large_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_large_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_xlarge_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_xlarge_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_xxlarge_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/1_albert_xxlarge_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_base_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_base_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_large_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_large_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_xlarge_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_xlarge_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_xxlarge_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/2_albert_xxlarge_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_base_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_base_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_large_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_large_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_xlarge_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_xlarge_v2_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_xxlarge_v1_32_384_2e-05_2_0",
#     "albert_args_train_results/3_albert_xxlarge_v2_32_384_2e-05_2_0",
# ]
#
# models = [models[x] for x in [2, 8, 10, 13, 16, 26, 27, 37, 44, 45, 52, 53]]

# vote1 models 15个模型 em: 89.994 f1: 92.252  ave: 91.123
models = [
    "args_train_results/1_electra_large_24_512_5e-05_2_2",
    "args_train_results/1_electra_large_32_384_5e-05_2_1",
    "args_train_results/1_electra_large_32_480_5e-05_2_1",
    "args_train_results/1_electra_large_32_512_5e-05_2_2",
    "args_train_results/2_electra_large_24_512_5e-05_2_1",
    "args_train_results/2_electra_large_32_384_5e-05_2_2",
    "args_train_results/2_electra_large_32_480_5e-05_2_2",
    "args_train_results/2_electra_large_32_512_5e-05_2_1",
    "atrlp_results/1",
    "atrlp_results/2",
    "atrlp_results/3",
    "atrlp_results/4",
    "atrlp_results/5",
    "atrlp_results/6",
    "atrlp_results/7",
    "atrlp_results/8",
    "atrlp_results/9",
    "atrlp_results/10",
    "lr_epoch_results/1e-05_2_2",
    "lr_epoch_results/1e-05_3_1",
    "lr_epoch_results/2e-05_2_3",
    "lr_epoch_results/2e-05_3_1",
    "lr_epoch_results/3.0000000000000004e-05_2_3",
    "lr_epoch_results/3.0000000000000004e-05_3_2",
    "lr_epoch_results/4e-05_2_2",
    "lr_epoch_results/4e-05_3_3",
    "lr_epoch_results/5e-05_2_2",
    "lr_epoch_results/5e-05_3_1",
    "lr_epoch_results/6e-05_2_1",
    "lr_epoch_results/6e-05_3_1",
    "albert_args_train_results/1_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xxlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xxlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xxlarge_v2_32_384_2e-05_2_0",
]

models = [models[x] for x in [1, 4, 5, 6, 8, 12, 21, 29, 32, 33, 36, 37, 39, 40, 41]]

pprint(models)
all_nbest = []
all_eval = []
all_odds = []
all_preds = []
for dire in [os.path.join(d) for d in models]:
    all_nbest.append(pickle.load(open(os.path.join(dire, 'eval_all_nbest.pkl'), 'rb')))
    all_eval.append(json.load(open(os.path.join(dire, 'squad_eval.json'), 'r', encoding='utf-8')))
    all_odds.append(json.load(open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))
    all_preds.append(json.load(open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))
qids = seq(all_preds[0].keys()).list()

# models_predictions = load_models_predictions(['atrlp_results'])

qid_answers = collections.OrderedDict()
qid_questions = collections.OrderedDict()
for article in json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data']:
    for paragraph in article["paragraphs"]:
        for qa in paragraph['qas']:
            _qid = qa['id']
            qid_answers[_qid] = qa['answers']
            qid_questions[_qid] = qa['question']


def vote1():
    """

    """
    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in qid_answers:
        bagging_preds[qid] = (seq([nbest[qid][0] for nbest in all_nbest])
                              .sorted(key=lambda x: x['probability'])
                              ).list()[-1]['text']
        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
    json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

    xargs = f"python eval.py dev-v2.0.json bagging_preds.json --na-prob-file bagging_odds.json"
    os.system(xargs)


def vote2():
    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in qid_answers:
        preds_scores = (seq(all_nbest)
                        .map(lambda x: x[qid][0])
                        .map(lambda x: (x['text'], x['probability']))
                        ).dict()
        compare = collections.defaultdict(lambda: 0.)
        for pred, score in preds_scores.items():
            compare[pred] += score
        compare = seq(compare.items()).sorted(lambda x: x[1]).reverse().list()
        bagging_preds[qid] = compare[0][0]

        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
    json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

    xargs = f"python eval.py dev-v2.0.json bagging_preds.json --na-prob-file bagging_odds.json"
    os.system(xargs)


def vote_with_post_processing():
    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    def post_process(question, candi, weight=1):
        question = question.lower()
        first_token = candi['text'].split()[0]
        th = 0.
        if "when" in question:
            if first_token in ['before', 'after', 'about', 'around', 'from', 'during']:
                candi['probability'] += th
        elif "where" in question:
            if first_token in ['in', 'at', 'on', 'behind', 'from', 'through', 'between', 'throughout']:
                candi['probability'] += th
        elif "whose" in question:
            if "'s" in candi['text']:
                candi['probability'] += th
        elif "which" in question:
            if first_token == "the":
                candi['probability'] += th
        candi['probability'] *= weight
        return candi

    cof = 0.2

    for qid in qid_answers:
        question = qid_questions[qid]
        post_process_candidates = (seq(zip(all_nbest, models))
                                   .map(lambda x: (x[0][qid], cof if 'lr_epoch_results' in x[1] else 1.))
                                   .map(lambda x: seq(x[0])
                                        .map(lambda y: post_process(question, y, x[1]))
                                        .list()
                                        )
                                   .flatten()
                                   ).list()
        preds_probs = collections.defaultdict(lambda: [])
        for pred in post_process_candidates:
            preds_probs[pred['text']].append(pred['probability'])
        for pred in post_process_candidates:
            preds_probs[pred['text']] = np.mean(preds_probs[pred['text']]).__float__()
        bagging_preds[qid] = (seq(preds_probs.items())
                              .sorted(lambda x: x[1])
                              .reverse()
                              .map(lambda x: x[0])
                              ).list()[0]
        bagging_odds[qid] = np.mean(
            [odds[qid] * cof if 'lr_epoch_results' in model else odds[qid] for odds, model in zip(all_odds, models)])

    r = main2(json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data'], bagging_preds, bagging_odds)
    print(f"{models}, {r}")


vote1()
# vote2()
# vote_with_post_processing()
