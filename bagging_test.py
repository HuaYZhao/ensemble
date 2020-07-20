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
import itertools
from sklearn import preprocessing
from eval import main2
from multiprocessing import Pool as ProcessPool

electra_albert_models = [
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
    "albert_args_train_results/1_albert_base_v1_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_base_v2_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_large_v1_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_large_v2_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/1_albert_xxlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_base_v1_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_base_v2_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_large_v1_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_large_v2_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/2_albert_xxlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_base_v1_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_base_v2_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_large_v1_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_large_v2_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xlarge_v2_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xxlarge_v1_32_384_2e-05_2_0",
    "albert_args_train_results/3_albert_xxlarge_v2_32_384_2e-05_2_0",
]

xlnet_models = [
    "xlnet_args_train_results/squad_large_32_1e-05_0",
    "xlnet_args_train_results/squad_large_32_1e-05_1",
    "xlnet_args_train_results/squad_large_32_1e-05_2",
    "xlnet_args_train_results/squad_large_32_3e-05_0",
    "xlnet_args_train_results/squad_large_32_3e-05_1",
    "xlnet_args_train_results/squad_large_32_3e-05_2",
    "xlnet_args_train_results/squad_large_48_3e-05_0",
    "xlnet_args_train_results/squad_large_48_3e-05_1",
    "xlnet_args_train_results/squad_large_48_3e-05_2",
    "xlnet_args_train_results/squad_large_48_5e-05_0",
    "xlnet_args_train_results/squad_large_48_5e-05_1",
    "xlnet_args_train_results/squad_large_48_5e-05_2",
]

c_models = [electra_albert_models[x] for x in [2, 3, 7, 10, 11, 16, 19, 21, 27, 35, 36, 44, 45, 52, 53]]

for xlnet in xlnet_models:
    print(xlnet)
    models = c_models + [xlnet]

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

    models_predictions = collections.OrderedDict()
    for d in models:
        try:
            prediction = collections.OrderedDict()
            prediction['squad_eval'] = json.load(
                open(os.path.join(d, 'squad_eval.json'), 'r', encoding='utf-8'))
            prediction['squad_null_odds'] = json.load(
                open(os.path.join(d, 'squad_null_odds.json'), 'r', encoding='utf-8'))
            prediction['squad_preds'] = json.load(open(os.path.join(d, 'squad_preds.json'), 'r', encoding='utf-8'))
            prediction['eval_all_nbest'] = pickle.load(open(os.path.join(d, 'eval_all_nbest.pkl'), 'rb'))
            # prediction['is_impossible'] = (seq(prediction['squad_null_odds'].items())
            #     .map(
            #     lambda x: [x[0], 1 if (x[1] > prediction['squad_eval']['best_exact_thresh']) else -1])
            # ).dict()
            models_predictions[f"{d}"] = prediction
        except:
            print(f"error at {d}")
            continue

    base_model_num = seq(models_predictions).filter(lambda x: 'base' in x).len()

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


    def get_model_cof(model_name):
        if "xlnet" in model_name:
            return 0.7
        else:
            return 1


    def vote2():
        """
        实验 base 与 large 之间的融合策略
        """
        bagging_preds = collections.OrderedDict()
        bagging_odds = collections.OrderedDict()

        for qid in qid_answers:
            bagging_preds[qid] = (seq(models_predictions.items())
                                  .map(lambda x: [get_model_cof(x[0]), x[1]['eval_all_nbest'][qid][0]])
                                  .map(lambda x: [x[1]['text'], x[0] * x[1]['probability']])
                                  .sorted(key=lambda x: x[1])
                                  .reverse()
                                  .map(lambda x: x[0])
                                  ).list()[0]
            bagging_odds[qid] = np.mean((seq(models_predictions.items())
                                         .map(lambda x: get_model_cof(x[0]) * x[1]['squad_null_odds'][qid])
                                         ).list())

        json.dump(bagging_preds, open('bagging_preds.json', 'w', encoding='utf-8'))
        json.dump(bagging_odds, open('bagging_odds.json', 'w', encoding='utf-8'))

        xargs = f"python eval.py dev-v2.0.json bagging_preds.json --na-prob-file bagging_odds.json"
        os.system(xargs)
        pass


    def vote_with_post_processing():
        bagging_preds = collections.OrderedDict()
        bagging_odds = collections.OrderedDict()

        def post_process(question, candi, weight=1):
            question = question.lower()
            if not candi['text']:
                return candi
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
                [odds[qid] * cof if 'lr_epoch_results' in model else odds[qid] for odds, model in
                 zip(all_odds, models)])

        r = main2(json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))['data'], bagging_preds, bagging_odds)
        print(f"{models}, {r}")


    vote2()
    # vote_with_post_processing()
