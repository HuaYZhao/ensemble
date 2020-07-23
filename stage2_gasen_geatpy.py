# -*- coding: utf-8 -*-
# @Time    : 2020/7/8 9:51
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import numpy as np
import geatpy as ea
from pprint import pprint
import os
import json
import pickle
import collections
import time
import typing
from functional import seq
from eval import compute_f1, compute_exact, normalize_answer, main2
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool

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
    # "args_train_pv_results/3_electra_large_32_512_2e-05_2_0",
    "args_train_pv_results/3_electra_large_32_512_5e-05_2_0",
    "args_train_pv_results/3_electra_large_48_256_6e-05_2_0",
]

models_predictions = collections.OrderedDict()
for d in models:
    try:
        prediction = collections.OrderedDict()
        if "albert" in d:
            prediction['squad_null_odds'] = json.load(
                open(os.path.join(d, 'squad_preds.json'), 'r', encoding='utf-8'))
        else:
            prediction['squad_null_odds'] = json.load(
                open(os.path.join(d, 'squad_null_odds.json'), 'r', encoding='utf-8'))
        models_predictions[f"{d}"] = prediction
    except:
        print(f"error at {d}")
        continue

dev = json.load(open('dev-v2.0.json', 'r', encoding='utf-8'))
qid_answers = collections.OrderedDict()
for article in dev['data']:
    for p in article['paragraphs']:
        for qa in p['qas']:
            qid = qa['id']
            gold_answers = [a['text'] for a in qa['answers']
                            if normalize_answer(a['text'])]
            if not gold_answers:
                # For unanswerable questions, only correct answer is empty string
                gold_answers = ['']
            qid_answers[qid] = gold_answers

base_preds = json.load(open("bagging_preds.json", 'r', encoding='utf-8'))
base_null_odds = json.load(open("bagging_odds.json", 'r', encoding='utf-8'))


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = len(models_predictions)  # 初始化Dim（决策变量维数）
        maxormins = [-1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # self.pool = ThreadPool(10)  # 设置池的大小
        num_cores = int(6)  # 获得计算机的核心数
        self.pool = ProcessPool(num_cores)  # 设置池的大小

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(zip(list(range(pop.sizes)), [Vars] * pop.sizes))
        pop.ObjV = np.vstack(self.pool.map(subAimFunc, args))


class MyAlgorithm(ea.moea_NSGA2_templet):

    def __init__(self, problem, population):
        ea.moea_NSGA2_templet.__init__(self, problem, population)
        self.best_trace = {}

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数
        gen = 0
        # ===========================准备进化============================
        population.initChrom()  # 初始化种群染色体矩阵
        self.call_aimFunc(population)  # 计算种群的目标函数值
        self.log(gen, population)
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        [levels, criLevel] = self.ndSort(population.ObjV, NIND, None, population.CV,
                                         self.problem.maxormins)  # 对NIND个个体进行非支配分层
        population.FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度
        # ===========================开始进化============================
        while self.terminated(population) == False:
            gen += 1
            # 选择个体参与进化
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 对选出的个体进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
            self.call_aimFunc(offspring)  # 求进化后个体的目标函数值
            self.log(gen, offspring)
            population = self.reinsertion(population, offspring, NIND)  # 重插入生成新一代种群
        self.best_trace = (seq(self.best_trace.items())
                           .sorted(key=lambda x: [-x[1], sum(json.loads(x[0]))])
                           ).dict()
        pprint(f"Best trace: {self.best_trace}")
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果

    def log(self, i, pop):
        best_score = np.max(pop.ObjV[:, 0])
        best_indexes = np.where(pop.ObjV[:, 0] == best_score)
        print(f"time: {time.time()}")
        for idx in best_indexes:
            model_num = sum(map(lambda x: x == 1, pop.Phen[idx, :][0]))
            phen = json.dumps(pop.Phen[idx, :].tolist()[0])
            if phen not in self.best_trace:
                self.best_trace[phen] = best_score
            print(
                f"Generation: {i}, model num #: {model_num} best score: {best_score}, best phen: {phen}")


def subAimFunc(args):
    assert models_predictions
    i = args[0]
    Vars = args[1]
    model_selections = Vars[i, :len(models_predictions)].tolist()

    # model_cofs = seq(Vars[i, len(models_predictions):]).map(lambda x: x / 10).list()

    def get_model_cof(model_name):
        return 1

    f2 = sum(model_selections) * 1e-5

    # Normalise weights
    # weights = preprocessing.normalize(np.reshape(indv.solution, (1, -1)), axis=1, norm='l1')
    indv_models_predictions = (seq(zip(*(models_predictions.items(), model_selections)))
                               .filter(lambda x: x[1] == 1)
                               .map(lambda x: x[0])
                               ).dict()

    # indv_models_cofs = seq(zip(model_cofs, model_selections)).filter(lambda x: x[1] == 1).map(lambda x: x[0]).list()
    if not indv_models_predictions:
        f1 = 0.
        return [f1, f2]

    ensemble_preds = collections.OrderedDict()
    ensemble_odds = collections.OrderedDict()
    for qid in qid_answers.keys():
        # preds_scores = (seq(indv_models_predictions.items())
        #                 .enumerate()
        #                 .map(lambda x: [get_model_cof(x[1][0]), x[1][1]['eval_all_nbest'][qid]])
        #                 .map(lambda x: [(y['text'], x[0] * y['probability']) for y in x[1]])
        #                 .flatten()
        #                 ).dict()
        # compare = collections.defaultdict(lambda: 0.)
        # for pred, score in preds_scores.items():
        #     compare[pred] += score
        # compare = seq(compare.items()).sorted(lambda x: x[1]).reverse().list()
        # ensemble_preds[qid] = compare[0][0]
        # ensemble_odds[qid] = np.mean((seq(indv_models_predictions.items())
        #                               .enumerate()
        #                               .map(lambda x: get_model_cof(x[1][0]) * x[1][1]['squad_null_odds'][qid])
        #                               ).list())
        ensemble_preds[qid] = base_preds[qid]
        ensemble_odds[qid] = np.mean((seq(indv_models_predictions.items())
                                      .map(lambda x: get_model_cof(x[0]) * x[1]['squad_null_odds'][qid])
                                      ).list()) + base_null_odds[qid]
    eval_r = main2(dev['data'], ensemble_preds, ensemble_odds)
    f1 = (eval_r['best_exact'] + eval_r['best_f1']) / 2
    return [f1, f2]


if __name__ == '__main__':
    """================================实例化问题对象==========================="""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'BG'  # 编码方式
    NIND = 12  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = MyAlgorithm(problem, population)  # 实例化一个算法模板对象

    myAlgorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 100  # 最大进化代数
    """==========================调用算法模板进行种群进化========================"""
    print("start GA process !")
    NDSet = myAlgorithm.run()  # 执行算法模板，得到帕累托最优解集NDSet
    NDSet.save()  # 把结果保存到文件中
    # 输出
    print('用时：%s 秒' % (myAlgorithm.passTime))
    print('非支配个体数：%s 个' % (NDSet.sizes))
    print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // myAlgorithm.passTime)))
