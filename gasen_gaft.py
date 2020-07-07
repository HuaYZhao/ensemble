# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 16:49
# @Author  : zhaohuayang
# @email   : zhaohuayang@myhexin.com
import os
import json
import pickle
import collections
import typing
from functional import seq
from eval import compute_f1, compute_exact, normalize_answer
from sklearn import preprocessing
import numpy as np
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

# Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.analysis import FitnessStore
from pprint import pprint


def load_models_predictions(dirs: typing.Any = './'):
    if dirs == './':
        dirs = [d for d in os.listdir(dirs) if d.endswith("results")]
    model_prediction = collections.OrderedDict()
    for d in sorted(dirs):
        for sub_d in sorted(os.listdir(d)):
            try:
                prediction = collections.OrderedDict()
                prediction['squad_eval'] = json.load(
                    open(os.path.join(d, sub_d, 'squad_eval.json'), 'r', encoding='utf-8'))
                prediction['squad_null_odds'] = json.load(
                    open(os.path.join(d, sub_d, 'squad_null_odds.json'), 'r', encoding='utf-8'))
                prediction['squad_preds'] = json.load(
                    open(os.path.join(d, sub_d, 'squad_preds.json'), 'r', encoding='utf-8'))
                prediction['eval_all_nbest'] = pickle.load(open(os.path.join(d, sub_d, 'eval_all_nbest.pkl'), 'rb'))
                prediction['is_impossible'] = (seq(prediction['squad_null_odds'].items())
                    .map(
                    lambda x: [x[0], 1 if (x[1] > prediction['squad_eval']['best_exact_thresh']) else -1])
                ).dict()
                model_prediction[f"{d}_{sub_d}"] = prediction
            except:
                continue

    pprint(f"Load Model Finished! Model keys are {model_prediction.keys()} !")
    return model_prediction


models_predictions = load_models_predictions()

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

indv_template = BinaryIndividual(ranges=[(0, 1)] * len(models_predictions), eps=0.001)
population = Population(indv_template=indv_template, size=50)
population.init()  # Initialize population with individuals.

# Use built-in operators here.
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation,
                  analysis=[FitnessStore])


# @engine.fitness_register
# def fitness(indv):
#     x, = indv.solution
#     return x + 10 * sin(5 * x).__float__() + 7 * cos(4 * x).__float__()


@engine.fitness_register
def ensemble_fitness(indv):
    assert models_predictions

    # Normalise weights
    weights = preprocessing.normalize(np.reshape(indv.solution, (1, -1)), axis=1, norm='l1')

    ensemble_preds = collections.OrderedDict()
    exact_scores = collections.OrderedDict()
    f1_scores = collections.OrderedDict()
    for qid in qid_answers.keys():
        best_pred = (seq(models_predictions.values())
                     .enumerate()
                     .map(lambda x: [x[0], x[1]['eval_all_nbest'][qid][0]])
                     .map(lambda x: [x[1]['text'], x[1]['probability'] * weights[0, x[0]]])
                     .sorted(key=lambda x: x[1])
                     .reverse()
                     .map(lambda x: x[0])
                     ).list()[0]
        is_impossible = (seq(models_predictions.values())
                         .enumerate()
                         .map(lambda x: x[1]['is_impossible'][qid] * weights[0, x[0]])
                         ).sum() > 0
        if is_impossible:
            ensemble_preds[qid] = ""
        else:
            ensemble_preds[qid] = best_pred
        exact_scores[qid] = max(compute_exact(a, ensemble_preds[qid]) for a in qid_answers[qid])
        f1_scores[qid] = max(compute_f1(a, ensemble_preds[qid]) for a in qid_answers[qid])

    return float((np.mean(list(exact_scores.values())) + np.mean(list(f1_scores.values()))) / 2)


@engine.analysis_register
class ConsoleOutput(OnTheFlyAnalysis):
    master_only = True
    interval = 1

    def register_step(self, g, population, engine):
        best_indv = population.best_indv(engine.fitness)
        msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.fmax)
        engine.logger.info(msg)


if '__main__' == __name__:
    engine.run(ng=100)
