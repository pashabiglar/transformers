#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
import pandas as pd
import numpy as np

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
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

    print_confusion_matrix(cm)
    print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

#read tsv predictions from sandeeps tensorflow code
#test_prediction_logits=pd.read_csv("data/5epochs_hugging_face_test_results_fevercrossdomain.txt",sep="\t",header=None)
test_prediction_logits=pd.read_csv("data/predictions_on_fnc_dev_with_sandeeps_personc1_vocab_studentteacher_10epochs_seed42.txt",sep="\t",header=None)
test_gold=pd.read_csv("data/fnc_dev_gold.tsv",sep="\t",header=None)


#why are the lengths different?
assert len(test_gold)==len(test_prediction_logits)

pred_labels=[]
gold_labels=[]

#assuming all rows match
index=0
for (pred,actual_row) in zip(test_prediction_logits.values,test_gold.values):
    index = index + 1
    if(index==1):
        continue
    #label_index=(np.argmax((pred).tolist()))
    label_string=pred[1]
    pred_labels.append(label_string)
    gold_labels.append(actual_row[1])

#assuming there is no corresponding prediction for 1st gold value/datapoint
# for index,predictions_row in enumerate(test_prediction_logits.values):
#     if index<25411:seed
#         label_index=(np.argmax(predictions_row.tolist()))
#         label_string=LABELS[label_index]
#         pred_labels.append(label_string)
#
#         gold_label=(test_gold.values[index+1][1])
#         gold_labels.append(gold_label)
#     else:
#         break

print(len(pred_labels))

assert len(pred_labels)==len(gold_labels)
report_score(gold_labels,pred_labels)