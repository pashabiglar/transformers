import torch
import os
from torch.utils.data.dataloader import DataLoader
import sys

def compare_dataloaders(dataset1, dataset2):
    models_differ = 0
    dataloaders1 = DataLoader(dataset1)
    dataloaders2 = DataLoader(dataset2)

    for i, data in enumerate(zip(dataloaders1, dataloaders2)):
        print(data)

    #     if torch.equal(key_item_1[1], key_item_2[1]):
    #         pass
    #     else:
    #         models_differ += 1
    #         if (key_item_1[0] == key_item_2[0]):
    #             print('Mismtach found at', key_item_1[0])
    #         else:
    #             raise Exception
    # if models_differ == 0:
    #     print('Models match perfectly! :)')
    # else:
    #     print("Esta madre. Models are different")

basedir="output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/"

#/home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/2/trained_model_at_end_of_epoch0_of_total_2.0epochs.pth


dataloader_path1=os.path.join(basedir,"1/training_data_at_epoch0_of_total_1epochs.pt")
dataloader_path2=os.path.join(basedir,"2/training_data_at_epoch0_of_total_2epochs.pt")


dataloader1 = torch.load(dataloader_path1,map_location=torch.device('cpu'))
dataloader2 = torch.load(dataloader_path2,map_location=torch.device('cpu'))

if dataloader1==dataloader2:
    print("dataloaders are same")
    sys.exit()

if dataloader1 is dataloader2:
    print("dataloaders are same")
    sys.exit()


compare_dataloaders(dataloader1, dataloader2)