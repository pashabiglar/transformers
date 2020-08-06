import torch
import os
import json
from transformers import TrainingArguments

# def get_optimizers(
#         self, num_training_steps: int
# ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
#     """
#     Setup the optimizer and the learning rate scheduler.
#
#     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
#     Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
#     """
#     if self.optimizers is not None:
#         return self.optimizers
#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": self.args.weight_decay,
#         },
#         {
#             "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
#     )
#     return optimizer, scheduler

# def deserialize_optimizers(model_path):
#     optimizer, scheduler = get_optimizers(num_training_steps=2)
#     if (
#             model_path is not None
#             and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
#             and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
#     ):
#         # Load in optimizer and scheduler states
#         optimizer.load_state_dict(
#             torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
#         )
#         scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))



import os
from dataclasses import dataclass, field
from typing import  Optional

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

def compare_models_bin(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                pass;
                #print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
    else:
        print("Esta madre. Models are different in compare_models_bin")


def compare_training_arguments(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1, model_2):
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

def compare_training_args(model_saved_as_bin2,model_saved_as_bin1):
    models_differ=0
    if(model_saved_as_bin2==model_saved_as_bin1):
        pass;
    else:
        models_differ+=1
    if models_differ == 0:
        print('Models match perfectly in compare_training_args! :)')
    else:
        print("Esta madre. Models are different in compare_training_args")



def compare_config_files(path1,path2):
    models_differ=0
    model_saved_as_bin1=_dict_from_json_file(path1)
    model_saved_as_bin2 = _dict_from_json_file(path2)
    if(model_saved_as_bin2==model_saved_as_bin1):
        pass;
    else:
        models_differ+=1
    if models_differ == 0:
        print('config files perfectly match! :)')
    else:
        print("Esta madre. config files are different ")


def _dict_from_json_file( json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

#assuming model was saved using torch.save_dict
def compare_models_saved_as_state_dicts(model_1, model_2):
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


basedir="/Users/mordor/research/huggingface_bert/mithun_scripts/output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/"

#basedir="output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/"

#/home/u11/mithunpaul/xdisk/huggingface_bert_fix_parallelism_per_epoch_issue/output/fever/fevercrossdomain/lex/figerspecific/bert-base-cased/128/2/
# trained_model_at_end_of_epoch0_of_total_2.0epochs.pth

#compare models that were saved using save_state_dictionary
# model_name_or_path1=os.path.join(basedir, "trained_model_at_end_of_epoch0_of_total_1.0epochs.pth")
# model_name_or_path2=os.path.join(basedir,"trained_model_at_end_of_epoch0_of_total_2.0epochs.pth")
# model1 = torch.load(model_name_or_path1, map_location=torch.device('cpu'))
# model2 = torch.load(model_name_or_path2,map_location=torch.device('cpu'))
# compare_models_saved_as_state_dicts(model_name_or_path1,model_name_or_path2)



#compare models that were saved using torch.save()
model_name_or_path1=os.path.join(basedir, "1/pytorch_model.bin")
model_name_or_path2=os.path.join(basedir,"2/pytorch_model.bin")
model1 = torch.load(model_name_or_path1, map_location=torch.device('cpu'))
model2 = torch.load(model_name_or_path2,map_location=torch.device('cpu'))
compare_models_bin(model1,model2)


# #to compare config files
model_name_or_path1=os.path.join(basedir, "1/config.json")
model_name_or_path2=os.path.join(basedir,"2/config.json")
compare_config_files(model_name_or_path1,model_name_or_path2)


#to compare training arg files
model_name_or_path1=os.path.join(basedir, "1/training_args.bin")
model_name_or_path2=os.path.join(basedir,"2/training_args.bin")
model1 = (torch.load(model_name_or_path1, map_location=torch.device('cpu')))
model2 = (torch.load(model_name_or_path2,map_location=torch.device('cpu')))



compare_training_args(model1,model2)

