# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import git
import numpy as np


from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    StudentTeacherTrainer,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
import math
from transformers.data.datasets import ParallelDataDataset,Read3DatasetsParallely


def get_git_info():
    repo = git.Repo(search_parent_directories=True)

    repo_sha=str(repo.head.object.hexsha),
    repo_short_sha= str(repo.git.rev_parse(repo_sha, short=6))

    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
        "repo_short_sha" :repo_short_sha
    }
    return repo_infos

logger = logging.getLogger(__name__)


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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    run_training(model_args, data_args, training_args)

def run_training(model_args, data_args, training_args):
    # Setup logging
    git_details=get_git_info()

    log_file_name=git_details['repo_short_sha']+"_"+(training_args.task_type)+"_"+(training_args.subtask_type1)+"_"+(training_args.subtask_type2)+"_"+str(model_args.model_name_or_path).replace("-","_")+"_"+data_args.task_name+".log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        filename=log_file_name,
        filemode='w'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)




    # Set seed
    set_seed(training_args.seed)

    try:

        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model_stu_teacher and tokenizer_lex
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model_stu_teacher & vocab.
  
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer_lex = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        force_download=True,
    )

    #when in student-teacher mode, you need two tokenizers, one for lexicalized data, and one for the delexicalized data
    # the regular tokenizer_lex will be used for lexicalized data and special one for delexicalized
    #delex tokenizer reads from specialized handcrafted vocabulary which understands symbols like personC1..rather symbols like C1,C2 etc
    #update@jan2021: when using multiple delexicalized dataset, we will use same delex vocabulary for all different types of delexicalization.
    #  this is because even though there are different tokens (like personC1 for oaner and actorc1 for figerspecific) ultimately we want it to be split into
    # person and c1- which any delex vocab will do
    tokenizer_delex = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        force_download=True,
        tokenizer_type="delex"
    )

    if (training_args.do_train_student_teacher == True):
        list_all_models = []
        for x in range(training_args.no_of_teacher_models):
            model_stu_teacher = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            )
            list_all_models.append(model_stu_teacher)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Get datasets

    # ParallelDataDataset is to be used in a student teacher model/architecture.
    # In this model teacher sees the lex data and student sees the delexicalized version of the same data
    # The one to one mapping is taken care of inside the ParallelDataDataset
    if (training_args.do_train_student_teacher == True):
        # the task type must be combined, not lex or delex. also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
        assert (training_args.task_type == "combined" or training_args.task_type=="2t1s")
        assert tokenizer_lex is not None
        assert tokenizer_delex is not None
        train_dataset = (
            Read3DatasetsParallely(args=data_args, tokenizer_lex=tokenizer_lex, tokenizer_delex=tokenizer_delex, data_type_1="lex", data_type_2="delex",
                                cache_dir=model_args.cache_dir) if training_args.do_train else None
        )
    else:
        if(training_args.task_type=="lex"):
            train_dataset = (
            GlueDataset(args=data_args, tokenizer=tokenizer_lex, task_type="lex", mode="train",
                        cache_dir=model_args.cache_dir)
            if training_args.do_train
            else None
            )
        else:
            if (training_args.task_type == "delex"):
                train_dataset = (
                    GlueDataset(args=data_args, tokenizer=tokenizer_delex, task_type="delex", mode="train",
                                cache_dir=model_args.cache_dir)
                    if training_args.do_train
                    else None
                )





    # in the student teacher mode we will keep the dev as in-domain dev delex partition. The goal here is to find how the
    # combined model_stu_teacher performs in a delexicalized dataset. This will serve as a verification point
    #to confirm the accuracy (we got 92.91% for fever delx in domain) if something goes wrong in the prediction phase below


    if (training_args.do_train_student_teacher == True):
        # the task type must be combined, not lex or delex. also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
        eval_dataset = (
            GlueDataset(args=data_args, tokenizer=tokenizer_delex, task_type="delex", mode="dev",
                        cache_dir=model_args.cache_dir)
            if training_args.do_eval
            else None
        )
    else:
        if (training_args.task_type == "lex"):
            eval_dataset = (
                GlueDataset(args=data_args, tokenizer=tokenizer_lex, task_type="lex", mode="dev",
                            cache_dir=model_args.cache_dir)
                if training_args.do_eval
                else None
            )
        else:
            if (training_args.task_type == "delex"):
                eval_dataset = (
                    GlueDataset(args=data_args, tokenizer=tokenizer_delex, task_type="delex", mode="dev",
                                cache_dir=model_args.cache_dir)
                    if training_args.do_eval
                    else None
                )




    if (training_args.do_train_student_teacher == True):
        # the task type must be combined, not lex or delex. also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
        # in the student teacher mode the evaluation always happens in the delex cross domain dev data. here we are loading it as the test partition so that we can keep track of
        # progress across epochs
        test_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer_delex, task_type="delex", mode="test", cache_dir=model_args.cache_dir)
            if training_args.do_predict
            else None
        )
    else:
        if (training_args.task_type == "lex"):
            test_dataset = (
                GlueDataset(data_args, tokenizer=tokenizer_lex, task_type="lex", mode="test",
                            cache_dir=model_args.cache_dir)
                if training_args.do_predict
                else None
            )

        else:
            if (training_args.task_type == "delex"):
                test_dataset = (
                    GlueDataset(data_args, tokenizer=tokenizer_delex, task_type="delex", mode="test",
                                cache_dir=model_args.cache_dir)
                    if training_args.do_predict
                    else None
                )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    dev_compute_metrics = build_compute_metrics_fn("feverindomain")
    test_compute_metrics = build_compute_metrics_fn("fevercrossdomain")

    if training_args.do_train_student_teacher:
        assert len(list_all_models)>1
        trainer = StudentTeacherTrainer(
            tokenizer_delex,
            tokenizer_lex,
            models=list_all_models,
            args=training_args,
            train_datasets={"combined": train_dataset},
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            eval_compute_metrics=dev_compute_metrics,
            test_compute_metrics=test_compute_metrics
        )
    else:
        trainer = Trainer(
            tokenizer_delex,
            tokenizer_lex,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            eval_compute_metrics=dev_compute_metrics,
            test_compute_metrics=test_compute_metrics
        )


    if training_args.do_train:
        dev_partition_evaluation_result=None
        test_partition_evaluation_result=None

        if (training_args.do_train_student_teacher == True):
            dev_partition_evaluation_result,test_partition_evaluation_result=trainer.train_multiple_teachers_1student(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        else:
            dev_partition_evaluation_result,test_partition_evaluation_result=trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )


        # For convenience, we also re-save the tokenizer_lex to the same directory,
        # so that you can share your model_stu_teacher easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer_lex.save_pretrained(training_args.output_dir)
        assert dev_partition_evaluation_result is not None
        assert test_partition_evaluation_result is not None
        return dev_partition_evaluation_result,test_partition_evaluation_result


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
