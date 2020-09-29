#In this code we go through predictions of 3 models and do a voting between 3 best.
'''
If(all 3 models agree) {
	use the predicted label

}

If (2 out of 3 models agree)

{

	use the predicted label

}

else {

	pick the label predicted with the highest confidence (after softmax)

}

'''
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

    #print_confusion_matrix(cm)
    #print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score




model1_predictions=pd.read_csv("predictions/predictions_on_test_partition_using_combined_trained_model_acc6921_2a528.txt", sep="\t", header=None)
model2_predictions=pd.read_csv("predictions/predictions_on_test_partition_using_delex_trained_model_de10f_54.04accuracy.txt", sep="\t", header=None)
model3_predictions=pd.read_csv("predictions/fnc_dev_gold.tsv", sep="\t", header=None)

# are the lengths different?
assert len(model3_predictions) == len(model1_predictions)
assert len(model3_predictions) == len(model2_predictions)
assert len(model1_predictions) == len(model2_predictions)

model1_predicted_labels_string=[]
model2_predicted_labels_string=[]
gold_labels=[]
model1_sf=[]
model2_sf=[]

#strip out labels from rest of the junk
for (mod1, mod2, mod3) in zip(model1_predictions.values, model2_predictions.values, model3_predictions.values):
    label_string=mod1[3]
    model1_predicted_labels_string.append(label_string)
    label_string = mod2[3]
    model2_predicted_labels_string.append(label_string)
    gold_labels.append(mod3[1])
    model1_sf.append(mod1[2])
    model2_sf.append(mod2[2])


assert len(model1_predicted_labels_string) == len(model2_predicted_labels_string) == len(gold_labels)
assert len(model1_sf) == len(model2_sf) == len(gold_labels)
assert not model1_sf[0] ==model2_sf[0]


#for some reason the softmaxes became a string
def convert_sf_string_to_lists(sf):
    list_as_floats=[]
    for x in sf.split(","):
        y=x.strip("[]")
        list_as_floats.append(float(y))
    return list_as_floats



def two_model_voting(model1_predicted_labels, model2_predicted_labels, model1_softmaxes, model2_softmaxes):
    assert len(model1_predicted_labels) == len(model2_predicted_labels)
    assert len(model1_softmaxes) == len(model2_softmaxes)

    predictions_post_voting=[]
    for index,(pred_model1, pred_model2,sf1,sf2) in enumerate(zip(model1_predicted_labels, model2_predicted_labels, model1_softmaxes, model2_softmaxes)):
        if (pred_model1==pred_model2):
            predictions_post_voting.append(pred_model1)
        else:
            #if the labels dont match, find who has higher confidence score
            sf1_list=convert_sf_string_to_lists(sf1)
            sf2_list = convert_sf_string_to_lists(sf2)
            highest_confidence_model1=max(sf1_list)
            highest_confidence_model2 = max(sf2_list)
            if highest_confidence_model1>highest_confidence_model2:
                predictions_post_voting.append(pred_model1)
            else:
                predictions_post_voting.append(pred_model2)

    return predictions_post_voting


def convert_labels_from_string_to_index(label_list):
    return [LABELS.index(label) for label in label_list]


def simple_accuracy(preds, gold):
    total_right=0
    for p,g in zip(preds,gold):
        if(p == g):
            total_right+=1
    return (total_right*100)/len(preds)

predictions_post_voting=two_model_voting(model1_predicted_labels_string, model2_predicted_labels_string, model1_sf, model2_sf)
fnc_score_lex=report_score(gold_labels, predictions_post_voting)
pred_labels_int=convert_labels_from_string_to_index(predictions_post_voting)
gold_labels_int=convert_labels_from_string_to_index(gold_labels)
accuracy=simple_accuracy(pred_labels_int, gold_labels_int)
print(f"accuracy={accuracy}")
