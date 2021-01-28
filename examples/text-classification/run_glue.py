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
    Trainer,
    TrainingArguments,
    StudentTeacherTrainer,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
import math
from transformers.data.datasets import ParallelDataDataset


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

    # Setup logging
    git_details=get_git_info()

    log_file_name=git_details['repo_short_sha']+"_"+(training_args.task_type)+"_"+(training_args.subtask_type)+"_"+data_args.task_name+".log"
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

    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    sys.exit()

    # Set seed
    set_seed(training_args.seed)

    try:

        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]

    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model_teacher and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model_teacher & vocab.
  
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        force_download=True,
        tokenizer_type=training_args.task_type
    )

    #when in student-teacher mode, you need two tokenizers, one for lexicalized data, and one for the delexicalized data
    # the regular tokenizer will be used for lexicalized data and special one for delexicalized
    if (training_args.do_train_1student_1teacher == True):
        tokenizer_delex = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            force_download=True,
            tokenizer_type="delex"
        )

    if (training_args.do_train_1student_1teacher == True):
        model_teacher = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        )
        model_student = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        )
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
    if (training_args.do_train_1student_1teacher == True):
        # the task type must be combined, not lex or delex. also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
        assert training_args.task_type == "combined"
        assert tokenizer is not None
        assert tokenizer_delex is not None
        train_dataset = (
            ParallelDataDataset(args=data_args, tokenizer_lex=tokenizer, tokenizer_delex=tokenizer_delex, data_type_1="lex", data_type_2="delex",
                                cache_dir=model_args.cache_dir) if training_args.do_train else None
        )
    else:
        if(training_args.task_type=="lex"):
            train_dataset = (
            GlueDataset(args=data_args, tokenizer=tokenizer, task_type="lex", mode="train",
                        cache_dir=model_args.cache_dir)
            if training_args.do_train
            else None
            )
        else:
            if (training_args.task_type == "delex"):
                train_dataset = (
                    GlueDataset(args=data_args, tokenizer=tokenizer, task_type="delex", mode="train",
                                cache_dir=model_args.cache_dir)
                    if training_args.do_train
                    else None
                )



    #save only at the end of each epoch
    #training_args.save_steps = math.floor((len(train_dataset) / training_args.per_device_train_batch_size) * training_args.num_train_epochs)

    # in the student teacher mode we will keep the dev as in-domain dev delex partition. The goal here is to find how the
    # combined model_teacher performs in a delexicalized dataset. This will serve as a verification point
    #to confirm the accuracy (we got 92.91% for fever delx in domain) if something goes wrong in the prediction phase below


    if (training_args.do_train_1student_1teacher == True):
        # the task type must be combined, not lex or delex. also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
        eval_dataset = (
            GlueDataset(args=data_args, tokenizer=tokenizer, task_type="delex", mode="dev",
                        cache_dir=model_args.cache_dir)
            if training_args.do_eval
            else None
        )
    else:
        if (training_args.task_type == "lex"):
            eval_dataset = (
                GlueDataset(args=data_args, tokenizer=tokenizer, task_type="lex", mode="dev",
                            cache_dir=model_args.cache_dir)
                if training_args.do_eval
                else None
            )
        else:
            if (training_args.task_type == "delex"):
                eval_dataset = (
                    GlueDataset(args=data_args, tokenizer=tokenizer, task_type="delex", mode="dev",
                                cache_dir=model_args.cache_dir)
                    if training_args.do_eval
                    else None
                )






    if (training_args.do_train_1student_1teacher == True):
        # the task type must be combined, not lex or delex. also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
        # in the student teacher mode the evaluation always happens in the delex cross domain dev data. here we are loading it as the test partition so that we can keep track of
        # progress across epochs
        test_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer, task_type="delex", mode="test", cache_dir=model_args.cache_dir)
            if training_args.do_predict
            else None
        )
    else:
        if (training_args.task_type == "lex"):
            test_dataset = (
                GlueDataset(data_args, tokenizer=tokenizer, task_type="lex", mode="test",
                            cache_dir=model_args.cache_dir)
                if training_args.do_predict
                else None
            )

        else:
            if (training_args.task_type == "delex"):
                test_dataset = (
                    GlueDataset(data_args, tokenizer=tokenizer, task_type="delex", mode="test",
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
    if training_args.do_train_1student_1teacher:
        trainer = StudentTeacherTrainer(
            models={"teacher": model_teacher, "student": model_student},
            args=training_args,
            train_datasets={"combined": train_dataset},
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            eval_compute_metrics=dev_compute_metrics,
            test_compute_metrics=test_compute_metrics
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            eval_compute_metrics=dev_compute_metrics,
            test_compute_metrics=test_compute_metrics
        )


    if training_args.do_train:
        if (training_args.do_train_1student_1teacher == True):
            trainer.train_1teacher_1student(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        else:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
        # trainer.save_model(training_args.output_dir)
        # # For convenience, we also re-save the tokenizer to the same directory,
        # # so that you can share your model_teacher easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)


    # Commenting since evaluation is now done at the end of each epoch. not at the end of all epochs
    #
    # # Evaluation
    # eval_results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate1 at the end of all epochs ***")
    #
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     eval_datasets = [eval_dataset]
    #     if data_args.task_name == "mnli":
    #         mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
    #         eval_datasets.append(
    #             GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    #         )
    #
    #     for eval_dataset in eval_datasets:
    #         #feverindomain compute metric has only accuracy while fever cross domain has both accuracy and fnc score
    #         trainer.eval_compute_metrics = build_compute_metrics_fn("feverindomain")
    #         eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    #
    #         output_eval_file = os.path.join(
    #             training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
    #         )
    #         if trainer.is_world_master():
    #             with open(output_eval_file, "w") as writer:
    #                 logger.info("***** dev results {} *****".format(eval_dataset.args.task_name))
    #                 for key, value in eval_result.items():
    #                     logger.info("  %s = %s", key, value)
    #                     writer.write("%s = %s\n" % (key, value))
    #
    #         eval_results.update(eval_result)
    #
    # # RUN Evaluation Again on test partition which is actually crossdomains dev partition
    # eval_results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate2 at the end of all epochs= evaluating on the test partition ***")
    #
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     eval_datasets = [test_dataset]
    #     if data_args.task_name == "mnli":
    #         mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
    #         eval_datasets.append(
    #             GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
    #         )
    #
    #     for eval_dataset in eval_datasets:
    #         trainer.eval_compute_metrics = build_compute_metrics_fn("fevercrossdomain")
    #         eval_result = trainer.evaluate(eval_dataset=eval_dataset)
    #
    #         output_eval_file = os.path.join(
    #             training_args.output_dir, f"test_partition_results_{eval_dataset.args.task_name}.txt"
    #         )
    #         if trainer.is_world_master():
    #             with open(output_eval_file, "w") as writer:
    #                 logger.info("***** fnc dev results {} *****".format(eval_dataset.args.task_name))
    #                 for key, value in eval_result.items():
    #                     logger.info("  %s = %s", key, value)
    #                     writer.write("%s = %s\n" % (key, value))
    #
    #         eval_results.update(eval_result)
    # if training_args.do_predict:
    #     logging.info("*** Test ***")
    #     test_datasets = [test_dataset]
    #     if data_args.task_name == "mnli":
    #         mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
    #         test_datasets.append(
    #             GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
    #         )
    #
    #     for test_dataset in test_datasets:
    #         predictions = trainer.predict(test_dataset=test_dataset).predictions
    #         if output_mode == "classification":
    #             predictions = np.argmax(predictions, axis=1)
    #
    #         output_test_file = os.path.join(
    #             training_args.output_dir, f"predictions_labels_on_test_partition_{test_dataset.args.task_name}.txt"
    #         )
    #         if trainer.is_world_master():
    #             with open(output_test_file, "w") as writer:
    #                 logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
    #                 writer.write("index\tprediction\n")
    #                 for index, item in enumerate(predictions):
    #                     if output_mode == "regression":
    #                         writer.write("%d\t%3.3f\n" % (index, item))
    #                     else:
    #                         item = test_dataset.get_labels()[item]
    #                         writer.write("%d\t%s\n" % (index, item))
    # return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
