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
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors,glue_convert_pair_examples_to_features,_glue_convert_examples_to_features,glue_convert_examples_from_list_of_datasets_to_features
from ..processors.utils import InputFeatures

from random import randrange
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
        limit_length: Optional[int] = None,
        task_type: Optional[str] = None,
        index_in: Optional[int] = 0,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
            remove_stop_words_in=False

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

        #note:right now (as of August 2020)its a hardcoded list of labels. this need to change based on what dataset you are using to tune on
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

            else:
                logger.info(f"found that no cache file exists. Creating features from dataset file at {args.data_dir}. value of mode is {mode}")
                examples=None

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                    logger.info(f"Done readign dev data")
                elif mode == Split.test:
                    examples = self.processor.get_test_examples_given_dataset_index(args.data_dir, index=index_in)
                    logger.info(f"Done readign test data")
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                    assert examples[00].label in label_list
                    logger.info(f"Done readign training data")
                if limit_length is not None:
                    examples = examples[:limit_length]
                logger.info(f"going to get into function glue_convert_examples_to_features")

                #finding all NER entities. this is needed in attention calculations for bert
                # this was used in finding attention weights allocated by bert across all layres and heads
                #spacy causing issues in hpc. commenting out temporarily on jan 2021 since i am doing only training now and dont need this

                # if(task_type == "lex"):
                #     all_ner={}
                #     for x in examples:
                #         combined=x.text_a+x.text_b
                #         #todo: replace spacy with processors ner tagger.
                #         doc = nlp(combined)
                #         for ent in doc.ents:
                #             all_ner[ent.text]=1
                #     self.ner_tags=all_ner
                assert examples is not None
                self.features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    remove_stop_words=remove_stop_words_in,
                    max_length=args.max_seq_length,
                    task=args.task_name,
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



class Read3DatasetsParallely(Dataset):
    """
    Same as GlueDataset, but here you can read 3 datasets together. For example you can read the lexicalixed
     2 delexicalized versions(each delexicalized differently) of the same datasets, with each data point corresponding
     to its equivalent in the other dataset. This is done as part of experiment on Jan 2021 when we are trying to see if
     #using multiple teachers with different perspectives does better than just one teacher.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        training_args,
        args: GlueDataTrainingArguments,
        tokenizer_lex: PreTrainedTokenizer,
        tokenizer_delex: PreTrainedTokenizer,
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
                mode.value, tokenizer_lex.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer_lex.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        #update@oct2020: when running student teacher first time after a long time, pass--overwrite_cache so that parallell dataset is created and tokenized.

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
                    #update: will use 3 teachers each having a different
                    list_all_datasets=[]
                    for index in range(training_args.total_no_of_models_including_student_and_its_teachers):
                        list_all_datasets.append(self.processor.get_train_examples_given_dataset_index(args.data_dir,index))

                    # assert both datasets are congruent
                    len_datasets=len(list_all_datasets[0])
                    # pick a random value and assert they match in label and guid with that of the first dataset
                    rand_index = randrange(0, len_datasets)
                    rand_label = list_all_datasets[0][rand_index].label
                    rand_guid = list_all_datasets[0][rand_index].guid

                    for each_dataset in list_all_datasets:
                        assert len(each_dataset) == len_datasets
                        assert each_dataset[rand_index].label==rand_label
                        assert each_dataset[rand_index].guid == rand_guid

                        if limit_length is not None:
                            each_dataset = each_dataset[:limit_length]

                #treate separately when we have only one dataset. multiple datasets need more parallel processing
                if(len(list_all_datasets))==1:
                    self.features = _glue_convert_examples_to_features(
                        list_all_datasets,
                        tokenizer_lex,
                        tokenizer_delex,
                        max_length=args.max_seq_length,
                        label_list=label_list,
                        output_mode=self.output_mode,
                    )

                else:

                    self.features = glue_convert_examples_from_list_of_datasets_to_features(
                        list_all_datasets,
                        tokenizer_lex,
                        tokenizer_delex,
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
        tokenizer_lex: PreTrainedTokenizer,
        tokenizer_delex: PreTrainedTokenizer,
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
                mode.value, tokenizer_lex.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer_lex.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.

        #note: when running student teacher first time a long time, pass--overwrite_cache so that parallell dataset is created and tokenized.

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
                    examples1 = self.processor.get_train_examples_set1(args.data_dir)
                    examples2 = self.processor.get_train_examples_set2(args.data_dir)

                    # assert both datasets are congruent
                    for index,(x, y) in enumerate(zip(examples1, examples2)):
                        assert x.label == y.label
                        assert x.guid == y.guid

                if limit_length is not None:
                    examples1 = examples1[:limit_length]
                    examples2 = examples2[:limit_length]
                self.features = glue_convert_pair_examples_to_features(
                    examples1,
                    examples2,
                    tokenizer_lex,
                    tokenizer_delex,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                    task=args.task_name
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
