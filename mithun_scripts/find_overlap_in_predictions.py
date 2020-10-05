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

    print_confusion_matrix(cm)
    #print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score




model1_predictions=pd.read_csv("predictions/predictions_on_test_partition_combined_wandbGraphNameStoicVoice1012_githubSha0f4e32_accuracy7404_fncs6262.txt", sep="\t", header=None)
model2_predictions=pd.read_csv("predictions/predictions_on_test_partition_using_lex_wandbgraphNameQueithaze806_eoch2_accuracy6750_fncscore6458.txt", sep="\t", header=None)
model3_predictions=pd.read_csv("predictions/predictions_on_test_partition_using_combined_trained_model_acc6921_2a528.txt", sep="\t", header=None)
test_gold=pd.read_csv("predictions/fnc_dev_gold.tsv",sep="\t",header=None)
""

# are the lengths different?
assert len(test_gold)==len(model1_predictions)
assert len(test_gold)==len(model2_predictions)
assert len(model1_predictions) == len(model2_predictions)

model1_labels=[]
delex_labels=[]
combined_labels=[]
gold_labels=[]
#
# # strip out labels from rest of the junk
# for (lex, delex, actual_row) in zip(model1_predictions.values, model2_predictions.values, model3_predictions.values,test_gold.values):
#     label_string = lex[3]
#     model1_labels.append(label_string)
#     label_string = delex[3]
#     delex_labels.append(label_string)
#     gold_labels.append(actual_row[1])
#
# assert len(model1_labels) == len(delex_labels) == len(gold_labels)


def find_two_label_overlap():


    fnc_score_lex=report_score(gold_labels, model1_labels)
    fnc_score_delex=report_score(gold_labels,delex_labels)



    #find how many mismatches between mod1 and mod2
    #basically: find how many did mod1 predict right, that mod2 guy missed- so at the end of the day we will have something to learn from mod1 or not.

    mismatches=0
    learnables=0
    correct_model1=0
    correct_model2=0
    correct_both=0
    count_both_missed_gold=0
    did_hit_gold=False
    total_count=0

    for index,(preds_model1, preds_model2,gold) in enumerate(zip(model1_labels, delex_labels, gold_labels)):
        total_count+=1
        if not (preds_model1==preds_model2):
            mismatches=mismatches+1
        if preds_model1 == gold and preds_model2 == gold:
            correct_both+=1
            did_hit_gold=True

        if preds_model1==gold:
            correct_model1+=1
            did_hit_gold=True
            if not (preds_model1 == preds_model2):
                learnables +=1
        if preds_model2 == gold:
            correct_model2+=1
            did_hit_gold = True

        if not ((preds_model1 == gold) or (preds_model2==gold)):
            count_both_missed_gold+=1




    mismatches_percentages=float(mismatches * 100 / len(gold_labels))
    percent_learnables=float(learnables)*100/float(len(gold_labels))
    accuracy_lex= float(correct_model1) * 100 / float(len(gold_labels))
    accuracy_delex= float(correct_model2) * 100 / float(len(gold_labels))


    print(f"correct_model1 count={correct_model1}")
    print(f"correct_model2 count={correct_model2}")
    print(f"fnc_score_delex ={(fnc_score_delex)}")
    print(f"fnc_score_lex ={(fnc_score_lex)}")
    print(f"gold_labels_count ={len(gold_labels)}")
    print(f"count_both_missed_gold ={(count_both_missed_gold)}")

    print(f"overall mismatches_count between mod1 and mod2={mismatches}")

    ######## everything in percentages wrt total data
    print(f"mismatches_percentages ={mismatches_percentages}")
    print(f"percent_learnables={percent_learnables}")
    print(f"percentage accuracy_lex={accuracy_lex}")
    print(f"percentage accuracy_delex={accuracy_delex}")
    print(f"out of {len(gold_labels)} count where model1 predicted correctly but model2 didnt={learnables}")

    print(f"**************details needed for venn diagram")
    print(f"out of {len(gold_labels)} how many both got right? {correct_both}")
    print(f"out of {len(gold_labels)} how many did only model1  get right ={correct_model1-correct_both}")
    print(f"out of {len(gold_labels)} how many did only model2 get right ={correct_model2-correct_both}")

    #(Ab, aB, AB)
    plt.figure(figsize=(4,4))
    venn2(subsets = (correct_model1-correct_both, correct_model2-correct_both,correct_both), set_labels = ('studentTeacher_wandbGraphNameStoicVoice1012_accuracy7404_fncs6262 ', 'lex_wandbgraphNameQueithaze806_accuracy6750_fncscore6458 '),set_colors=('red', 'blue'), alpha = 0.7)
    plt.title("Overlap in cross domain predictions between \n 1 student teacher delex model and \n 1 lex alone trained \n  model  ")
    plt.show()



#get labels and softmaxes into its own lists
#column order in raw data: index	 gold	prediction_logits	 prediction_label	plain_text
def get_separate_lists_of_each_column_for_3_model_ensemble():
    for (mod1, mod2, mod3, gold) in zip(model1_predictions.values, model2_predictions.values, model3_predictions.values, test_gold.values):
        model1_labels.append(mod1[3])
        delex_labels.append(mod2[3])
        combined_labels.append(mod3[3])
        gold_labels.append(gold[1])


#get labels and softmaxes into its own lists
#column order in raw data: index	 gold	prediction_logits	 prediction_label	plain_text
def get_separate_lists_of_each_column_for_2_model_ensemble():
    for (mod1, mod2, gold) in zip(model1_predictions.values, model2_predictions.values, test_gold.values):
        model1_labels.append(mod1[3])
        delex_labels.append(mod2[3])
        gold_labels.append(gold[1])


def find_3_model_overlap():
    # (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    how_many_only_lex_got_right = 0  # Abc
    how_many_only_delex_got_right = 0  # aBc
    how_many_both_lex_delex_together_got_right = 0  # ABc
    how_many_only_combined_model_got_right = 0  # abC
    how_many_both_lex_and_combined_models_together_got_right = 0  # AbC
    how_many_both_delex_and_combined_models_together_got_right = 0  # aBC
    how_many_all3_got_right = 0  # ABC

    no_body_got_right = 0


    for index, (pred_lex, pred_delex, pred_combined,gold) in enumerate(zip(model1_labels, delex_labels, combined_labels, gold_labels)):
        flag_some_body_got_right = False
        if pred_lex == gold:
            how_many_only_lex_got_right += 1
            flag_some_body_got_right=True
        if pred_delex == gold:
            how_many_only_delex_got_right +=1
            flag_some_body_got_right=True
        if (pred_lex==pred_delex==gold):
            how_many_both_lex_delex_together_got_right+=1
            flag_some_body_got_right=True
        if pred_combined == gold:
            how_many_only_combined_model_got_right += 1
            flag_some_body_got_right=True
        if (pred_lex == pred_combined==gold):
            how_many_both_lex_and_combined_models_together_got_right += 1
            flag_some_body_got_right=True
        if (pred_delex == pred_combined==gold):
            how_many_both_delex_and_combined_models_together_got_right += 1
            flag_some_body_got_right=True
        if (pred_lex==pred_delex==pred_combined==gold):
            how_many_all3_got_right+=1
            flag_some_body_got_right=True
        if  (flag_some_body_got_right==False):
            no_body_got_right+=1

    return    how_many_only_lex_got_right,how_many_only_delex_got_right ,\
              how_many_both_lex_delex_together_got_right ,\
              how_many_only_combined_model_got_right ,\
              how_many_both_lex_and_combined_models_together_got_right ,\
              how_many_both_delex_and_combined_models_together_got_right ,\
              how_many_all3_got_right,\
            no_body_got_right

def print_outputs():
    print(f"**************details needed for venn diagram")
    print(f"out of {len(gold_labels)} how many both got right? {correct_both}")
    print(
        f"out of {len(gold_labels)} how many did only model1 (student teacher trained model) get right ={correct_lex-correct_both}")
    print(f"out of {len(gold_labels)} how many did only model2 get right ={correct_delex-correct_both}")

def draw_plots_3sets(Abc, aBc, ABc, abC, AbC, aBC, ABC):
    plt.figure(figsize=(4, 4))
    venn3(subsets=(Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels=('Lex', 'Delex', 'Student_teacher_trained'),
          set_colors=('red', 'blue',"yellow"), alpha=0.7)


    plt.title(
        "Overlap in cross domain predictions between \n lex, delex and student-teacher trained delex model")
    plt.show()

def verify_total_count(Abc, aBc, ABc, abC, AbC, aBC, ABC,no_body_got_right):
    #n(A) + n(B) + n(C) – n(A∩B) – n(B∩C) – n(C∩A) + n(A∩B∩C)
    total=Abc+aBc+abC-ABc-aBC-AbC+ABC
    assert total+no_body_got_right==len(gold_labels)

get_separate_lists_of_each_column_for_2_model_ensemble()
find_two_label_overlap()

# get_separate_lists_of_each_column_for_3_model_ensemble()
# assert len(model1_labels)==len(delex_labels)==len(combined_labels)==len(gold_labels)
# Abc, aBc, ABc, abC, AbC, aBC, ABC,no_body_got_right=find_3_model_overlap()
# verify_total_count(Abc, aBc, ABc, abC, AbC, aBC, ABC,no_body_got_right)
# draw_plots_3sets(Abc, aBc, ABc, abC, AbC, aBC, ABC,)