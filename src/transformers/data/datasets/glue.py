import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union
import sys
import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_bart import BartTokenizer, BartTokenizerFast
from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors,glue_convert_pair_examples_to_features
from ..processors.utils import InputFeatures


logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,

        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        task_type: Optional[str] = None,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,


    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        logger.info(f"value of cahced features file is {cached_features_file}")
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        logger.info(
            f"args.overwrite_cache {args.overwrite_cache}")

        logger.info(
            f"os.path.exists(cached_features_file) {os.path.exists(cached_features_file)}")

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
                sys.exit(1)
            else:
                logger.info(f"found that no cache file exists. Creating features from dataset file at {args.data_dir}. value of mode is {mode}")
                sys.exit(1)

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                    logger.info(f"Done readign dev data")
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                    logger.info(f"Done readign test data")
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                    logger.info(f"Done readign training data")
                if limit_length is not None:
                    examples = examples[:limit_length]
                logger.info(f"going to get into function glue_convert_examples_to_features")
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                logger.info(f"done with features. going to save features to cached features file whose value is {cached_features_file}")
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list



class ParallelDataDataset(Dataset):
    """
    Same as GlueDataset, but here you can read two datasets together. For example you can read the lexicalixed
    and delexicalized version of the same datasets, with each data point corresponding to its equivalent in the other dataset.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,

        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        data_type_1: Optional[str] = None,
        data_type_2: Optional[str] = None,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,


    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")
                #data_dir = os.path.join(args.data_dir, data_type_1)
                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    #when using parallel datasets get two features of examples and pass it to glue_convert_pair_examples_to_features
                    #which in turn creates features and combines them both
                    #data_dir1 = os.path.join(args.data_dir, data_type_1)
                    examples1 = self.processor.get_train_examples_set1(args.data_dir)
                    #data_dir2 = os.path.join(args.data_dir, data_type_2)
                    examples2 = self.processor.get_train_examples_set2(args.data_dir)
                if limit_length is not None:
                    examples1 = examples1[:limit_length]
                    examples2 = examples2[:limit_length]
                self.features = glue_convert_pair_examples_to_features(
                    examples1,
                    examples2,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list
