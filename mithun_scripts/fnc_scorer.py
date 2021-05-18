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

    gold_label_spread = {}

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        gold_label_spread[g]= gold_label_spread.get(g, 0) + 1
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm, gold_label_spread

def convert_labels_from_string_to_index(label_list):

    return [LABELS.index(label) for label in label_list]



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

def simple_accuracy(preds, gold):
    total_right=0
    for p,g in zip(preds,gold):
        if(p == g):
            total_right+=1
    return (total_right*100)/len(preds)

def report_score(actual,predicted):
    score,cm,gold_label_spread  = score_submission(actual,predicted)
    best_score, _,gold_label_spread = score_submission(actual,actual)
    print_gold_label_distribution(gold_label_spread)
    print_confusion_matrix(cm)
    print("FNC Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

#read tsv predictions from sandeeps tensorflow code
test_prediction_logits=pd.read_csv("predictions/predictions_on_test_partition_3e310f.txt",sep="\t",header=None)
test_gold=pd.read_csv("predictions/fnc_dev_gold.tsv",sep="\t",header=None)


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
    label_string=pred[3]
    pred_labels.append(label_string)
    gold_labels.append(actual_row[1])


def print_gold_label_distribution(gold_label_spread):
    for x in gold_label_spread.items():
        print(x)

print(len(pred_labels))

assert len(pred_labels)==len(gold_labels)
report_score(gold_labels,pred_labels)
pred_labels_int=convert_labels_from_string_to_index(pred_labels)
gold_labels_int=convert_labels_from_string_to_index(gold_labels)
accuracy=simple_accuracy(pred_labels_int,gold_labels_int)
print(f"accuracy={accuracy}")
