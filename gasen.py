# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 13:56
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
import GA


def load_models_predictions(dirs: typing.Any = './'):
    if dirs == './':
        dirs = [d for d in os.listdir(dirs) if d.endswith("results")]
    model_prediction = collections.OrderedDict()
    for d in dirs:
        for sub_d in os.listdir(d):
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

    return model_prediction


def ensemble_fitness(weights, models_predictions, qid_answers):
    assert models_predictions
    fitness = []

    # Normalise weights
    weights = preprocessing.normalize(weights, axis=1, norm='l1')

    for i in range(len(weights)):

        ensemble_preds = collections.OrderedDict()
        exact_scores = collections.OrderedDict()
        f1_scores = collections.OrderedDict()
        for qid in qid_answers.keys():
            best_pred = (seq(models_predictions.values())
                         .enumerate()
                         .map(lambda x: [x[0], x[1]['eval_all_nbest'][qid][0]])
                         .map(lambda x: [x[1]['text'], x[1]['probability'] * weights[i, x[0]]])
                         .sorted(key=lambda x: x[1])
                         .reverse()
                         .map(lambda x: x[0])
                         ).list()[0]
            is_impossible = (seq(models_predictions.values())
                             .enumerate()
                             .map(lambda x: x[1]['is_impossible'][qid] * weights[i, x[0]])
                             ).sum() > 0
            if is_impossible:
                ensemble_preds[qid] = ""
            else:
                ensemble_preds[qid] = best_pred
            exact_scores[qid] = max(compute_exact(a, ensemble_preds[qid]) for a in qid_answers[qid])
            f1_scores[qid] = max(compute_f1(a, ensemble_preds[qid]) for a in qid_answers[qid])

        fitness.append(
            -1. * float((np.mean(list(exact_scores.values())) + np.mean(list(f1_scores.values()))) / 2)
        )
    return fitness


def main():
    models_predictions = load_models_predictions(['atrlp_results'])
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

    # Create objective function
    objective_function = lambda w: ensemble_fitness(w, models_predictions, qid_answers)

    # Set Genetic Algorithm parameters
    sol_per_pop = 8
    num_parents_mating = 4

    # Defining population size
    pop_size = (sol_per_pop, len(models_predictions))
    # Creating the initial population
    new_population = np.random.uniform(low=0, high=1, size=pop_size)
    # print(new_population)

    num_generations = 100

    for generation in range(num_generations):
        print("Generation: ", generation)
        # Measuring the fitness of each chromosome in the population
        fitness = GA.cal_pop_fitness(objective_function, new_population)
        print(f"Best fitness: {-1 * np.min(fitness).__float__()}")

        # Selecting the best parents in the population for mating
        parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)

        # Generating next generation using crossover
        offspring_crossover = GA.crossover(parents,
                                           offspring_size=(pop_size[0] - parents.shape[0], len(models_predictions)))

        # Adding some variations to the offspring using mutation
        offspring_mutation = GA.mutation(offspring_crossover)

        # Creating the new population based on the parents and offspring
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        # The best result in the current iteration

    # Get the best solution after all generations
    fitness = GA.cal_pop_fitness(objective_function, new_population)
    # Return the index of that solution and corresponding best fitness
    best_match_idx = np.where(fitness == np.min(fitness))
    print(best_match_idx)
    print(fitness)


if __name__ == '__main__':
    main()
