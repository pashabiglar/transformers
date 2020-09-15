# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_renamed(preds, labels):
        acc = simple_accuracy(preds, labels)
        return {
            "in_domain_acc": acc
        }

    def acc_and_fnc_score(preds, labels):
        acc = simple_accuracy(preds, labels)
        cm, f1 = calculate_fnc_score(labels, preds)
        return {
            "cross_domain_acc": acc,
            "cross_domain_fnc_score": f1,
            "confusion matrix": cm
        }

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"mnli/acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"mnli-mm/acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "feverindomain":
            return {"acc": acc_renamed(preds, labels)}
        elif task_name == "fevercrossdomain":
            return {"acc": acc_and_fnc_score(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
    #Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
    #Original credit - @bgalbraith
    import pandas as pd
    import numpy as np

    LABELS = ['disagree', 'agree', 'discuss', 'unrelated']
    LABELS_RELATED = ['unrelated','related']
    RELATED = LABELS[0:3]

    def score_submission(gold_labels, test_labels):
        score = 0.0
        cm = [[0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]

        for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
            g_stance, t_stance = g, t
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 'unrelated':
                    score += 0.50
            if g_stance in RELATED and t_stance in RELATED:
                score += 0.25

            cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

        return score, cm


    def print_confusion_matrix(cm):
        lines = []
        header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
        line_len = len(header)
        lines.append("-"*line_len)
        lines.append(header)
        lines.append("-"*line_len)

        hit = 0
        total = 0
        for i, row in enumerate(cm):
            hit += row[i]
            total += sum(row)
            lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                       *row))
            lines.append("-"*line_len)
        print('\n'.join(lines))

    def report_score(actual,predicted):
        score,cm = score_submission(actual,predicted)
        best_score, _ = score_submission(actual,actual)
        return cm,score/best_score

    def calculate_fnc_score(actual,predicted):
        print("inside calculate_fnc_score")
        actual=[LABELS[x] for x in actual]
        predicted = [LABELS[x] for x in predicted]
        cm,score=report_score(actual,predicted)
        return [cm,score]