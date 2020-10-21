'''
based on the paper: Statistical Significance Tests for Machine Translation Evaluation
Philipp Koehn
https://www.aclweb.org/anthology/W04-3250.pdf

steps for boostrap resampling on one dataset with one model:
1. get all the predictions on fnc-dev (9068 data points) by a given model (say you pick a model which has accuracy of 74%)
2. from this randomly pick 300 data points (and their corresponding gold)
for 1 to 1000:
    3. from this 300 pick 300 data points, with replacement
    4. calculate accuracy, note it down
    5. repeat 1000 times

6. plot all these accuracies
7. check if 95% of the time you get an accuracy of 74% or more




steps for boostrap resampling on a pair of models:


'''

import pandas as pd
import random


LABELS = ['disagree', 'agree', 'discuss', 'unrelated']

delex_predictions=pd.read_csv("for_bootstrap/fnc_dev/predictions_on_fnc_dev_by_delex_trained_model_acc55point3.txt", sep="\t")
lex_predictions=pd.read_csv("for_bootstrap/fnc_dev/predictions_on_fnc_dev_by_lex_trained_model_acc70point21.txt", sep="\t")
student_teacher_predictions=pd.read_csv("for_bootstrap/fnc_dev/predictions_on_fnc_dev_by_student_teacher_trained_model_acc74point74.txt", sep="\t")
gold=pd.read_csv("for_bootstrap/fnc_dev/fnc_dev_gold.tsv", sep="\t")


# are the lengths different?
assert len(gold) == len(delex_predictions)
assert len(gold) == len(lex_predictions)
assert len(delex_predictions) == len(lex_predictions)
assert len(delex_predictions) == len(student_teacher_predictions)

delex_model_predicted_labels_string=[]
lex_model_predicted_labels_string=[]
student_teacher_model_predicted_labels_string=[]
gold_labels=[]
model1_sf=[]
model2_sf=[]
model3_sf=[]

#get labels and softmaxes into its own lists
#column order in raw data: index	 gold	prediction_logits	 prediction_label	plain_text
for (mod1, mod2, mod3) in zip(delex_predictions.values, lex_predictions.values, student_teacher_predictions.values):
    delex_model_predicted_labels_string.append(mod1[3])
    lex_model_predicted_labels_string.append(mod2[3])
    student_teacher_model_predicted_labels_string.append(mod3[3])
    gold_labels.append(mod3[1])
    model1_sf.append(mod1[2])
    model2_sf.append(mod2[2])
    model3_sf.append(mod3[2])


def convert_labels_from_string_to_index(label_list):
    return [LABELS.index(label) for label in label_list]

def simple_accuracy(preds, gold):
    total_right=0
    for p,g in zip(preds,gold):
        if(p == g):
            total_right+=1
    return (total_right*100)/len(preds)

total=len(student_teacher_model_predicted_labels_string)

#set a seed for reproducability. comment this in final calculations
random.seed(3)

#pick 300 random numbers between 0 and len
rand_list=[]
for x in range (0,300):
    n=random.randint(0,total)
    rand_list.append(n)

assert len(rand_list)==300

# for each such element in the list, pick the corresponding data point and its corresponding gold label
pred_rand_list=[]
gold_rand_list=[]
for x in rand_list:
    assert student_teacher_model_predicted_labels_string[x] is not None
    assert gold_labels[x] is not None
    pred_rand_list.append(student_teacher_model_predicted_labels_string[x])
    gold_rand_list.append(gold_labels[x])



pred_labels_int=convert_labels_from_string_to_index(pred_rand_list)
gold_labels_int=convert_labels_from_string_to_index(gold_rand_list)
accuracy=simple_accuracy(pred_labels_int, gold_labels_int)
print(f"accuracy={accuracy}")

