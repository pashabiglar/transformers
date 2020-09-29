#Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py
#Original credit - @bgalbraith
import pandas as pd
import numpy as np

#Import libraries
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
#matplotlib inline

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

    #print_confusion_matrix(cm)
    #print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score




lex_predictions=pd.read_csv("predictions/predictions_on_test_partition_using_combined_trained_model_acc6921_2a528.txt", sep="\t", header=None)
delex_predictions=pd.read_csv("predictions/predictions_on_test_partition_using_delex_trained_model_de10f_54.04accuracy.txt", sep="\t", header=None)
test_gold=pd.read_csv("predictions/fnc_dev_gold.tsv",sep="\t",header=None)

# are the lengths different?
assert len(test_gold)==len(lex_predictions)
assert len(test_gold)==len(delex_predictions)
assert len(lex_predictions)==len(delex_predictions)

lex_labels=[]
delex_labels=[]
gold_labels=[]


#strip out labels from rest of the junk
for (lex,delex,actual_row) in zip(lex_predictions.values,delex_predictions.values,test_gold.values):
    label_string=lex[3]
    lex_labels.append(label_string)
    label_string = delex[3]
    delex_labels.append(label_string)
    gold_labels.append(actual_row[1])


assert len(lex_labels)==len(delex_labels)==len(gold_labels)



fnc_score_lex=report_score(gold_labels,lex_labels)
fnc_score_delex=report_score(gold_labels,delex_labels)



#find how many mismatches between mod1 and mod2
#basically: find how many did mod1 predict right, that mod2 guy missed- so at the end of the day we will have something to learn from mod1 or not.

mismatches=0
learnables=0
correct_lex=0
correct_delex=0
correct_both=0
for index,(pred_lex, pred_delex,gold) in enumerate(zip(lex_labels,delex_labels,gold_labels)):
    if not (pred_lex==pred_delex):
        mismatches=mismatches+1
    if pred_lex == gold and pred_delex == gold:
        correct_both+=1

    if pred_lex==gold:
        correct_lex+=1
        if not (pred_lex == pred_delex):
            learnables +=1
    if pred_delex == gold:
        correct_delex+=1



mismatches_percentages=float(mismatches * 100 / len(gold_labels))
percent_learnables=float(learnables)*100/float(len(gold_labels))
accuracy_lex= float(correct_lex) * 100 / float(len(gold_labels))
accuracy_delex= float(correct_delex) * 100 / float(len(gold_labels))


print(f"correct_lex count={correct_lex}")
print(f"correct_delex count={correct_delex}")
print(f"fnc_score_delex ={(fnc_score_delex)}")
print(f"fnc_score_lex ={(fnc_score_lex)}")
print(f"gold_labels_count ={len(gold_labels)}")

print(f"overall mismatches_count between mod1 and mod2={mismatches}")

######## everything in percentages wrt total data
print(f"mismatches_percentages ={mismatches_percentages}")
print(f"percent_learnables={percent_learnables}")
print(f"percentage accuracy_lex={accuracy_lex}")
print(f"percentage accuracy_delex={accuracy_delex}")
print(f"out of {len(gold_labels)} count where model1 predicted correctly but model2 didnt={learnables}")

print(f"**************details needed for venn diagram")
print(f"out of {len(gold_labels)} how many both got right? {correct_both}")
print(f"out of {len(gold_labels)} how many did only model1 (student teacher trained model) get right ={correct_lex-correct_both}")
print(f"out of {len(gold_labels)} how many did only model2 get right ={correct_delex-correct_both}")

plt.figure(figsize=(4,4))
venn2(subsets = (2036, 660,4240), set_labels = ('student teacher trained model ', 'stand alone delex trained model'),set_colors=('red', 'blue'), alpha = 0.7)
plt.title("Overlap in cross domain predictions between \n student-teacher trained delex model and stand alone delex model")
plt.show()

