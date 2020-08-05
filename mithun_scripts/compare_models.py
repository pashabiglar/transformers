import torch
import os

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
    else:
        print("Esta madre. Models are different")

basedir="../output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/"
basedir=""
#/home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/2/trained_model_at_end_of_epoch0_of_total_2.0epochs.pth

path1=os.path.join(basedir,"trained_model_at_end_of_epoch0_of_total_1.0epochs.pth")
path2=os.path.join(basedir,"trained_model_at_end_of_epoch0_of_total_2.0epochs.pth")

model1 = torch.load(path1,map_location=torch.device('cpu'))
model2 = torch.load(path2,map_location=torch.device('cpu'))

compare_models(model1,model2)