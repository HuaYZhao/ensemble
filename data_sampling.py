# -*- coding: utf-8 -*-
# @Time    : 2020/7/2 16:40
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import random
import json
import pickle
import collections
from functional import seq
from copy import deepcopy


class Bootstrap:
    __instanceList: list

    def __init__(self, instanceList: list, seed: int):
        """
        A constructor of Bootstrap class which takes a sample an array of instances and a seed number, then creates a
        bootstrap sample using this seed as random number.
        PARAMETERS
        ----------
        instanceList : list
            Original sample
        seed : int
            Random number to create boostrap sample
        """
        random.seed(seed)
        N = len(instanceList)
        self.__instanceList = []
        for i in range(N):
            self.__instanceList.append(instanceList[random.randint(0, N - 1)])

    def getSample(self) -> list:
        """
        getSample returns the produced bootstrap sample.
        RETURNS
        -------
        list
            Produced bootstrap sample
        """
        return self.__instanceList


qid_answers = collections.OrderedDict()
for article in json.load(open('train-v2.0.json', 'r', encoding='utf-8'))['data']:
    for paragraph in article["paragraphs"]:
        for qa in paragraph['qas']:
            qid = qa['id']
            qid_answers[qid] = qa['answers']
qids = seq(qid_answers.keys()).list()


def generate_difficult_distinguish_train_data():
    atrlp_dir = ['1', '3', '9']

    all_nbest = []
    all_eval = []
    all_odds = []
    all_preds = []
    for dire in [os.path.join('./atrlp_results', d, 'train') for d in atrlp_dir]:
        all_nbest.append(pickle.load(open(os.path.join(dire, 'eval_all_nbest.pkl'), 'rb')))
        all_eval.append(json.load(open(os.path.join(dire, 'squad_eval.json'), 'r', encoding='utf-8')))
        all_odds.append(json.load(open(os.path.join(dire, 'squad_null_odds.json'), 'r', encoding='utf-8')))
        all_preds.append(json.load(open(os.path.join(dire, 'squad_preds.json'), 'r', encoding='utf-8')))

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

    difficult_distinguish_qids = has_answer_diff_qids + no_answer_diff_qids
    print(len(difficult_distinguish_qids))
    train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))
    for article in train['data']:
        for paragraph in article["paragraphs"]:
            new_qas = []
            for qa in paragraph['qas']:
                qid = qa['id']
                if qid in difficult_distinguish_qids:
                    new_qas.append(qa)
            paragraph['qas'] = new_qas

    json.dump(train, open('difficult_train.json', 'w', encoding='utf-8'))
    print("dump difficult distinguish train data! ")


def generate_bootstrap_train_data(k=3):
    for i in range(k):
        bootstrap_qids = collections.Counter(Bootstrap(qids, i * 1024 + 1).getSample())
        train = json.load(open('train-v2.0.json', 'r', encoding='utf-8'))
        for article in train['data']:
            for paragraph in article["paragraphs"]:
                new_qas = []
                for qa in paragraph['qas']:
                    qid = qa['id']
                    if qid in bootstrap_qids:
                        for _ in range(bootstrap_qids[qid]):
                            new_qas.append(qa)
                paragraph['qas'] = new_qas

        json.dump(train, open(f'bootstrap_train_{i}.json', 'w', encoding='utf-8'))
        print(f"dump difficult bootstrap train {i} data! ")


if __name__ == '__main__':
    # generate_difficult_distinguish_train_data()
    generate_bootstrap_train_data()
