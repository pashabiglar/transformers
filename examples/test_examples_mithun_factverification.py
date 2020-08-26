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
import unittest
from unittest.mock import patch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from typing import Optional
from transformers import GlueDataTrainingArguments as DataTrainingArguments


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in ["text-generation", "text-classification", "language-modeling", "question-answering"]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_generation
    import run_glue
    import run_language_modeling
    import run_squad


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


class ExamplesTests(unittest.TestCase):
    def test_run_glue(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = f"""
            run_glue.py
            --model_name_or_path bert-base-uncased 
            --task_name fevercrossdomain 
            --do_train --do_eval --do_predict 
            --data_dir ../src/transformers/data/datasets/fever/fevercrossdomain/lex/figerspecific --max_seq_length 128 
            --per_device_eval_batch_size=16 --per_device_train_batch_size=16 --learning_rate 1e-5 --num_train_epochs 1 
            --output_dir ./output/fever/fevercrossdomain/lex/figerspecific/bert-base-uncased/128/ --overwrite_output_dir 
            --weight_decay 0.01 --adam_epsilon 1e-6 --evaluate_during_training --task_type lex --subtask_type figerspecific --machine_to_run_on laptop
            """.split()

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))


        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=testargs[1:len(testargs)])


        with patch.object(sys, "argv", testargs):
            #Note: assumption here that the test will be run for 1 epoch only. ELse have to return the best dev and test partition scores
            dev_partition_evaluation_result,test_partition_evaluation_result = run_glue.main()
            accuracy_dev_partition = dev_partition_evaluation_result['eval_acc']
            fnc_score_test_partition = test_partition_evaluation_result['eval_acc']['fnc_score']
            accuracy_test_partition = test_partition_evaluation_result['eval_acc']['acc']

            # check if the training meets minimum accuracy. note that in laptop we run on a toy data set of size 16 and
            # in hpc (high performance computing server) we test on 100 data points. so the threshold accuracy to check
            # is different in each case
            if(training_args.machine_to_run_on=="laptop"):
                self.assertGreaterEqual(fnc_score_test_partition, 0.025)
                self.assertGreaterEqual(accuracy_test_partition, 0.0625)
                self.assertGreaterEqual(accuracy_dev_partition, 0.0625)
            else:
                if (training_args.machine_to_run_on == "hpc"):
                    self.assertGreaterEqual(fnc_score_test_partition, 0.75)
                    self.assertGreaterEqual(accuracy_test_partition, 0.75)
                    self.assertGreaterEqual(accuracy_dev_partition, 0.75)

            #todo: run for 100 data points. find good accuracy and fnc score numbers and change from 0.75. do the same for  multiple epochs

    def test_run_language_modeling(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)
        # TODO: switch to smaller model like sshleifer/tiny-distilroberta-ba`se

        testargs = """
            run_language_modeling.py
            --model_name_or_path distilroberta-base
            --model_type roberta
            --mlm
            --line_by_line
            --train_data_file ./tests/fixtures/sample_text.txt
            --eval_data_file ./tests/fixtures/sample_text.txt
            --output_dir ./tests/fixtures/tests_samples/temp_dir
            --overwrite_output_dir
            --do_train
            --do_eval
            --num_train_epochs=1
            --no_cuda
            """.split()
        with patch.object(sys, "argv", testargs):
            result = run_language_modeling.main()
            self.assertLess(result["perplexity"], 35)

    def test_run_squad(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = """
            run_squad.py
            --model_type=distilbert
            --model_name_or_path=sshleifer/tiny-distilbert-base-cased-distilled-squad
            --data_dir=./tests/fixtures/tests_samples/SQUAD
            --output_dir=./tests/fixtures/tests_samples/temp_dir
            --max_steps=10
            --warmup_steps=2
            --do_train
            --do_eval
            --version_2_with_negative
            --learning_rate=2e-4
            --per_gpu_train_batch_size=2
            --per_gpu_eval_batch_size=1
            --overwrite_output_dir
            --seed=42
        """.split()
        with patch.object(sys, "argv", testargs):
            result = run_squad.main()
            self.assertGreaterEqual(result["f1"], 25)
            self.assertGreaterEqual(result["exact"], 21)

    def test_generation(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        testargs = ["run_generation.py", "--prompt=Hello", "--length=10", "--seed=42"]
        model_type, model_name = ("--model_type=gpt2", "--model_name_or_path=sshleifer/tiny-gpt2")
        with patch.object(sys, "argv", testargs + [model_type, model_name]):
            result = run_generation.main()
            self.assertGreaterEqual(len(result[0]), 10)
