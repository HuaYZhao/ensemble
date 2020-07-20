# -*- coding: utf-8 -*-
# @Time    : 2020/7/14 11:16
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import pickle
import json
import collections
import shutil


def eval_a_model_output(dire):
    all_nbest = pickle.load(open(os.path.join(dire, "eval_all_nbest.pkl"), 'rb'))
    squad_null_odds = json.load(open(os.path.join(dire, "squad_null_odds.json"), 'r', encoding='utf-8'))
    squad_preds = collections.OrderedDict()
    for qid in squad_null_odds:
        squad_preds[qid] = all_nbest[qid][0]['text']
    shutil.move(os.path.join(dire, "squad_preds.json"), os.path.join(dire, "origin_squad_preds.json"))
    json.dump(squad_preds, open(os.path.join(dire, "squad_preds.json"), 'w', encoding='utf-8'))

    xargs = f"""python3.6 eval.py dev-v2.0.json {os.path.join(dire, "squad_preds.json")} -n {os.path.join(dire, "squad_null_odds.json")} -o {os.path.join(dire, "squad_eval.json")}"""
    os.system(xargs)


if __name__ == '__main__':
    all_model_path = "xlnet_args_train_results"
    for model in os.listdir(all_model_path):
        model_path = os.path.join(all_model_path, model)
        eval_a_model_output(model_path)
