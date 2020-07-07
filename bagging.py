# -*- coding: utf-8 -*-
# @Time    : 2020/7/1 11:09
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import pickle
import json
import collections
import numpy as np
from functional import seq
import itertools
from sklearn.linear_model import LinearRegression

all_atrlp_dirs = list(itertools.combinations([str(i) for i in range(1, 11)], 3))
atrlp_dir = ['1', '3', '9']
# for atrlp_dir in all_atrlp_dirs:
assert len(atrlp_dir)
print(f"now bagging atrlp_dir is {atrlp_dir}")

all_nbest = []
all_eval = []
all_odds = []
all_preds = []
for dire in [os.path.join('./atrlp_results', d) for d in atrlp_dir]:
    all_nbest.append(pickle.load(open(os.path.join(dire, 'eval_all_nbest.pkl'), 'rb')))
    all_eval.append(json.load(open(os.path.join(dire, 'squad_eval.json'), 'r', encoding='utf-8')))
    all_odds.append(json.load(open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))
    all_preds.append(json.load(open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))

qids = seq(all_preds[0].keys()).list()

qid_answers = collections.OrderedDict()
for article in json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data']:
    for paragraph in article["paragraphs"]:
        for qa in paragraph['qas']:
            qid = qa['id']
            qid_answers[qid] = qa['answers']


def vote1():
    """

    """
    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in qids:
        bagging_preds[qid] = (seq([nbest[qid][0] for nbest in all_nbest])
                              .sorted(key=lambda x: x['probability'])
                              ).list()[-1]['text']
        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
    json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

    xargs = f"python eval.py dev-v2.0.json bagging_preds.json --na-prob-file bagging_odds.json"
    os.system(xargs)

    not_exact_qids = (seq(bagging_preds.items())
                      .filter(lambda x: x[1] not in [y['text'] for y in qid_answers[x[0]]] and qid_answers[x[0]])
                      .map(lambda x: x[0])
                      ).list()

    analyse1 = (seq(qid_answers)
                .filter(lambda x: x in not_exact_qids)
                .map(lambda x: [y[x] for y in all_preds] + [qid_answers[x]] + [bagging_preds[x]])
                ).list()

    has_answer_diff_qids = (seq(qid_answers.items())
        .filter(lambda x: x[1])
        .map(lambda x: x[0])
        .filter(
        lambda x: len(collections.Counter([nbest[x][0]['text'] for nbest in all_nbest])) != 1)
    ).list()

    no_answer_diff_qids = (seq(qid_answers.items())
        .filter_not(lambda x: x[1])
        .map(lambda x: x[0])
        .filter_not(
        lambda x: all(all_odds[i][x] > all_eval[i]['best_exact_thresh'] for i in range(len(atrlp_dir))))
    ).list()


def vote2():
    """
    odds
    """
    bagging_preds = collections.OrderedDict()
    threshes = np.array([eva['best_exact_thresh'] for eva in all_eval])
    for qid in qids:
        if bool(sum([odd[qid] for odd in all_odds] > threshes) < np.ceil(len(atrlp_dir) / 2)):
            bagging_preds[qid] = (seq([nbest[qid][0] for nbest in all_nbest])
                                  .sorted(key=lambda x: x['probability'])
                                  ).list()[-1]['text']
        else:
            bagging_preds[qid] = ""
    json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
    xargs = f"python eval.py dev-v2.0.json bagging_preds.json "
    os.system(xargs)


def vote3(rank=False):
    bagging_preds = collections.OrderedDict()
    bagging_odds = collections.OrderedDict()

    for qid in qids:
        pred_d = collections.defaultdict(lambda: [])
        if rank:
            for nbest in all_nbest:
                for pred, v in zip(nbest[qid], seq(np.arange(0, 1, 1 / len(nbest[qid]))).reverse()):
                    pred['probability'] = v
        for pred in seq([nbest[qid] for nbest in all_nbest]).flatten():
            pred_d[pred['text']].append(pred['probability'])
        for k in pred_d:
            pred_d[k] = float(np.mean(pred_d[k]))
        bagging_preds[qid] = seq(pred_d.items()).sorted(key=lambda x: x[1], reverse=True).list()[0][0]

        bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

    json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
    json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

    xargs = f"python eval.py dev-v2.0.json bagging_preds.json --na-prob-file bagging_odds.json"
    os.system(xargs)


vote1()
# vote3(rank=True)
# vote2()
