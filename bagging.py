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
from sklearn import preprocessing
from eval import main2
from multiprocessing import Pool as ProcessPool

# from gasen_gaft import load_models_predictions
model_dirs = ['atrlp_results', 'bootstrap_results']

all_models = [os.path.join(d, dd) for d in model_dirs for dd in os.listdir(d)]
all_models_combinations = list(itertools.combinations(all_models, 7))


# atrlp_dir = ['1', '3', '9']
def main(models):
    assert len(models)

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
            lambda x: all(all_odds[i][x] > all_eval[i]['best_exact_thresh'] for i in range(len(models))))
        ).list()

    def vote2():
        """
        odds
        """
        bagging_preds = collections.OrderedDict()
        threshes = np.array([eva['best_exact_thresh'] for eva in all_eval])
        for qid in qid_answers:
            if bool(sum([odd[qid] for odd in all_odds] > threshes) < np.ceil(len(models) / 2)):
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

        for qid in qid_answers:
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

        for qid in qid_answers:
            question = qid_questions[qid]
            post_process_candidates = (seq(zip(all_nbest, models))
                                       .map(lambda x: (x[0][qid], 0.2 if 'bootstrap' in x[1] else 1))
                                       .map(lambda x: seq(x[0])
                                            .map(lambda y: post_process(question, y, x[1]))
                                            .list()
                                            )
                                       .flatten()
                                       ).list()
            preds_probs = collections.defaultdict(lambda: 0.)
            for pred in post_process_candidates:
                preds_probs[pred['text']] += pred['probability']
            bagging_preds[qid] = (seq(preds_probs.items())
                                  .sorted(lambda x: x[1])
                                  .reverse()
                                  .map(lambda x: x[0])
                                  ).list()[0]
            bagging_odds[qid] = np.mean([odds[qid] for odds in all_odds])

        r = main2(json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data'], bagging_preds, bagging_odds)
        print(f"{models}, {r}")

    # vote1()
    # vote3(rank=True)
    # vote2()
    vote_with_post_processing()


if __name__ == '__main__':
    pool = ProcessPool(6)  # 设置池的大小
    pool.map(main, all_models_combinations)
