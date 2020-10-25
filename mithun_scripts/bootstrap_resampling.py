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

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import scipy.stats as stats
import pandas as pd
import random

NO_OF_RUNS=1000
NO_OF_DATAPOINTS_PER_BATCH=700
LABELS = ['disagree', 'agree', 'discuss', 'unrelated']

#set a seed for reproducability. comment this in final calculations
np.random.seed(34345)
random.seed(334534)


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

def pvalue_101(mu, sigma, samp_size, samp_mean=0, deltam=0):
    np.random.seed(1234)
    s1 = np.random.normal(mu, sigma, samp_size)
    if samp_mean > 0:
        #print(len(s1[s1>samp_mean]))
        outliers = float(len(s1[s1>samp_mean])*100)/float(len(s1))
        print('Percentage of numbers larger than {} is {}%'.format(samp_mean, outliers))
    if deltam == 0:
        deltam = abs(mu-samp_mean)
    if deltam > 0 :
        outliers = (float(len(s1[s1>(mu+deltam)]))
                    +float(len(s1[s1<(mu-deltam)])))*100.0/float(len(s1))
        print('Percentage of numbers further than the population mean of {} by +/-{} is {}%'.format(mu, deltam, outliers))

    fig, ax = plt.subplots(figsize=(8,8))
    fig.suptitle('Normal Distribution: population_mean={}'.format(mu) )
    plt.hist(s1)
    plt.axvline(x=mu+deltam, color='red')
    plt.axvline(x=mu-deltam, color='green')
    plt.show()

def convert_labels_from_string_to_index(label_list):
    return [LABELS.index(label) for label in label_list]

def simple_accuracy(preds, gold):
    total_right=0
    for p,g in zip(preds,gold):
        if(p == g):
            total_right+=1
    return (total_right*100)/len(preds)

total=len(student_teacher_model_predicted_labels_string)


def per_distribution_bootstrap(distribution_to_test_with,gold):
    list_of_all_accuracies_across_all_runs=[]

    list_of_all_indices=[*range(0,total)]
    for x in range (0,NO_OF_RUNS):
        # pick NO_OF_DATAPOINTS_PER_BATCH random elements between 0 and len -with replacement
        rand_list=random.choices(list_of_all_indices,k=NO_OF_DATAPOINTS_PER_BATCH)
        assert len(rand_list)==NO_OF_DATAPOINTS_PER_BATCH

        # for each such element in the list, pick the corresponding data point and its corresponding gold label
        pred_rand_list=[]
        gold_rand_list=[]
        for x in rand_list:
            assert distribution_to_test_with[x] is not None
            assert gold[x] is not None
            pred_rand_list.append(distribution_to_test_with[x])
            gold_rand_list.append(gold[x])
        #now calculate accuracy for this batch of 300 data points
        pred_labels_int=convert_labels_from_string_to_index(pred_rand_list)
        gold_labels_int=convert_labels_from_string_to_index(gold_rand_list)
        accuracy=simple_accuracy(pred_labels_int, gold_labels_int)
        #print(f"accuracy={accuracy}")
        list_of_all_accuracies_across_all_runs.append(accuracy)
    return list_of_all_accuracies_across_all_runs


def scatter_plot_given_two_distributions(distribution1, distribution2, xaxis):
    plt.title('My title')
    plt.xlabel('categories')
    plt.ylabel('values')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(distribution1,xaxis,  c='b', marker="*", label='distribution')
    ax1.scatter(distribution2, xaxis, c='r', marker="8", label='second')
    #plt.plot(y, x, 'o', color='blue');
    plt.show()

def find_which_system_is_better(list1,list2):
    assert len(list1)==len(list2)
    count_x_higher=0
    for index,(x,y) in enumerate(zip(list1,list2)):
        if x>y:
            count_x_higher+=1
    print(f"out of total of {len(list1)} data points {count_x_higher} times list1 is higher than list 2. "
          f"i.e {count_x_higher*100/len(list1)} percentage ")


pred_labels_studentteacher_int=convert_labels_from_string_to_index(student_teacher_model_predicted_labels_string)
gold_labels_int=convert_labels_from_string_to_index(gold_labels)
actual_accuracy_student_teacher=simple_accuracy(pred_labels_studentteacher_int, gold_labels_int)
print(f"actual_accuracy_student_teacher={actual_accuracy_student_teacher}")

pred_labels_lex_int=convert_labels_from_string_to_index(lex_model_predicted_labels_string)
actual_accuracy_lex=simple_accuracy(pred_labels_lex_int, gold_labels_int)
print(f"actual_accuracy_lex={actual_accuracy_lex}")

accuracy_list_student_teacher= per_distribution_bootstrap(student_teacher_model_predicted_labels_string,gold_labels)
accuracy_list_lex= per_distribution_bootstrap(lex_model_predicted_labels_string,gold_labels)
find_which_system_is_better(accuracy_list_student_teacher,accuracy_list_lex)


assert len(accuracy_list_student_teacher)==len(accuracy_list_lex)==NO_OF_RUNS

scatter_plot_given_two_distributions(accuracy_list_student_teacher, accuracy_list_lex, np.arange(1, NO_OF_RUNS + 1))

# average=sum(accuracy_list)/len(accuracy_list)
# print(f"average of given sample={average}")
# pvalue_101(74.74,4.0,NO_OF_RUNS,average)
