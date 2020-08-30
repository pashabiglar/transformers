# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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

from dataclasses import dataclass, field
import argparse
import logging
import os
import sys
from unittest.mock import patch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from typing import Optional
from transformers import GlueDataTrainingArguments as DataTrainingArguments
import git


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in ["text-generation", "../text-classification", "language-modeling", "question-answering"]
]
sys.path.extend(SRC_DIRS)

if SRC_DIRS is not None:

    import run_glue

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()

def get_setup_file():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f

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


def test_run_glue(name):
        # Setup logging


        name_split = name.split()
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=name_split)

        git_details = get_git_info()

        log_file_name = git_details['repo_short_sha'] + "_" + (training_args.task_type) + "_" + (
            training_args.subtask_type) + "_" + str(model_args.model_name_or_path).replace("-",
                                                                                           "_") + "_" + data_args.task_name + ".log"
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

        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        dev_partition_evaluation_result,test_partition_evaluation_result = run_glue.run_training( model_args, data_args, training_args)
        accuracy_dev_partition = dev_partition_evaluation_result['eval_acc']
        fnc_score_test_partition = test_partition_evaluation_result['eval_acc']['fnc_score']
        accuracy_test_partition = test_partition_evaluation_result['eval_acc']['acc']
        logger.info(f"value of accuracy_dev_partition={accuracy_dev_partition}")
        logger.info(f"value of fnc_score_test_partition={fnc_score_test_partition}")
        logger.info(f"value of accuracy_test_partition={accuracy_test_partition}")
        # check if the training meets minimum accuracy. note that in laptop we run on a toy data set of size 16 and
        # in hpc (high performance computing server) we test on 100 data points. so the threshold accuracy to check
        # is different in each case
        #(training_args.task_type)+"_"+(training_args.subtask_type)+"_"+str(model_args.model_name_or_path)
        test_case_encountered=False
        if(training_args.machine_to_run_on=="laptop"):
            if(training_args.task_type=="lex"):
                if (training_args.subtask_type=="figerspecific"):
                    if (model_args.model_name_or_path=="bert-base-uncased"):
                        assert fnc_score_test_partition==0.025
                        assert accuracy_test_partition == 0.0625
                        assert accuracy_dev_partition == 0.0625

                        test_case_encountered=True
                    else:
                        if (model_args.model_name_or_path == "bert-base-cased"):
                            assert fnc_score_test_partition == 0.725
                            assert accuracy_test_partition == 0.75
                            assert accuracy_dev_partition == 0.375


                            test_case_encountered = True
            else:
                if (training_args.task_type == "delex"):
                    if (training_args.subtask_type == "figerspecific"):
                        if (model_args.model_name_or_path == "bert-base-uncased"):
                            assert fnc_score_test_partition == 0.1
                            assert accuracy_test_partition == 0.25
                            assert accuracy_dev_partition == 0.125



                            test_case_encountered = True
                        else:
                            if (model_args.model_name_or_path == "bert-base-cased"):
                                assert fnc_score_test_partition == 0.225
                                assert accuracy_test_partition == 0.5
                                assert accuracy_dev_partition == 0.125


                                test_case_encountered = True
        else:
            if (training_args.machine_to_run_on == "hpc"):
                if (training_args.task_type == "lex"):
                    if (training_args.subtask_type == "figerspecific"):
                        if (model_args.model_name_or_path == "bert-base-uncased"):
                            assert fnc_score_test_partition == 0.025
                            assert accuracy_test_partition == 0.0625
                            assert accuracy_dev_partition == 0.0625

                            test_case_encountered = True
                        else:
                            if (model_args.model_name_or_path == "bert-base-cased"):

                                #exact value has way too many precision points
                                assert fnc_score_test_partition > 0.57
                                assert fnc_score_test_partition < 0.58

                                assert accuracy_test_partition > 0.6
                                assert accuracy_test_partition < 0.7

                                assert accuracy_dev_partition > 0.3
                                assert accuracy_dev_partition < 0.4

                                test_case_encountered = True
                else:
                    if (training_args.task_type == "delex"):
                        if (training_args.subtask_type == "figerspecific"):
                            if (model_args.model_name_or_path == "bert-base-uncased"):
                                assert fnc_score_test_partition == 0.1
                                assert accuracy_test_partition == 0.25
                                assert accuracy_dev_partition == 0.125

                                test_case_encountered = True
                            else:
                                if (model_args.model_name_or_path == "bert-base-cased"):
                                    assert fnc_score_test_partition == 0.1
                                    assert accuracy_test_partition == 0.25
                                    assert accuracy_dev_partition == 0.125

                                    test_case_encountered = True


            assert test_case_encountered is True


            logger.info("done with fact verification related testing . going to exit")
            sys.exit(1)
#             #todo: run for 100 data points. find good accuracy and fnc score numbers and change from 0.75. do the same for  multiple epochs
