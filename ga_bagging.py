# -*- coding: utf-8 -*-
# @Time    : 2020/7/7 10:59
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
from gasen_gaft import load_models_predictions
from sklearn import preprocessing
import numpy as np
import json
import os
import collections
from functional import seq

models_predictions = load_models_predictions()
cof = [0.96875, 0.9609375, 0.283203125, 0.552734375, 0.958984375, 0.302734375, 0.91796875, 0.333984375, 0.044921875,
       0.2265625, 0.755859375, 0.607421875, 0.23046875, 0.134765625, 0.720703125, 0.44140625, 0.091796875, 0.509765625,
       0.6875, 0.740234375, 0.3984375, 0.251953125, 0.90625, 0.88671875, 0.0859375, 0.2421875, 0.70703125, 0.666015625,
       0.388671875, 0.349609375, 0.8203125, 0.798828125, 0.6875, 0.95703125, 0.451171875, 0.07421875, 0.05859375,
       0.3984375, 0.35546875, 0.01171875, 0.56640625, 0.25390625, 0.896484375, 0.41796875, 0.08203125, 0.3046875,
       0.287109375, 0.552734375, 0.01171875, 0.216796875]

weights = preprocessing.normalize(np.reshape(cof, (1, -1)), axis=1, norm='l1')[0]

dev = json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))

all_qids = []
for article in dev['data']:
    for p in article['paragraphs']:
        for qa in p['qas']:
            qid = qa['id']
            all_qids.append(qid)

models_predictions = (seq(zip(*(models_predictions.items(), weights)))
                      .filter(lambda x: x[1] > 0.5 / len(cof))
                      .map(lambda x: x[0])
                      ).dict()
ga_ensemble_preds = collections.OrderedDict()
ga_ensemble_odds = collections.OrderedDict()
for qid in all_qids:
    ga_ensemble_preds[qid] = \
        (seq([nbest[qid][0] for nbest in [v['eval_all_nbest'] for v in models_predictions.values()]])
         .sorted(key=lambda x: x['probability'])
         ).list()[-1]['text']
    ga_ensemble_odds[qid] = np.mean([odds[qid] for odds in [v['squad_null_odds'] for v in models_predictions.values()]])

json.dump(ga_ensemble_preds, open('ga_ensemble_preds.json', 'w', encoding='utf-8'))
json.dump(ga_ensemble_odds, open('ga_ensemble_odds.json', 'w', encoding='utf-8'))

xargs = f"python eval.py dev-v2.0.json ga_ensemble_preds.json --na-prob-file ga_ensemble_odds.json"
os.system(xargs)
