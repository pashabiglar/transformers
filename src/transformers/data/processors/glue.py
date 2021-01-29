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
""" GLUE processors and helpers """
import string
import logging
import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from tqdm import tqdm
from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from .utils import DataProcessor, InputExample, InputFeatures
import nltk
#from stop_words import get_stop_words
from nltk.corpus import stopwords


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    remove_stop_words,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,

):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        logger.info(f"found that is_tf_available is true")
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _glue_convert_examples_to_features(
        examples, tokenizer, remove_stop_words,max_length=max_length, task=task, label_list=label_list, output_mode=output_mode)

def glue_convert_pair_examples_to_features(
    examples1: Union[List[InputExample], "tf.data.Dataset"],
    examples2: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer_lex: PreTrainedTokenizer,
    tokenizer_delex: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data (where data comes in pairs/parallel datasets) file into a list of ``InputFeatures``
    note


    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer_lex: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples1, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples1, tokenizer_lex, max_length=max_length, task=task)

    return _glue_convert_pair_examples_to_features(
        examples1,examples2, tokenizer_lex=tokenizer_lex, tokenizer_delex=tokenizer_delex, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )




def glue_convert_examples_from_list_of_datasets_to_features(
    all_datasets,
    tokenizer_lex: PreTrainedTokenizer,
    tokenizer_delex: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data (where data comes in pairs/parallel datasets) file into a list of ``InputFeatures``
    note


    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer_lex: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(all_datasets[0], tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(all_datasets[0], tokenizer_lex, max_length=max_length, task=task)

    return _glue_convert_list_of_example_pairs_to_features(
        all_datasets, tokenizer_lex=tokenizer_lex, tokenizer_delex=tokenizer_delex, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = ["input_ids"] + tokenizer.model_input_names

        return tf.data.Dataset.from_generator(
            gen,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )


def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    remove_stop_words,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None
):
    logger.info("inside function _glue_convert_examples_to_features")
    if max_length is None:
        max_length = tokenizer.max_len
    logger.info(f"fvalue of task is {task}")
    if task is not None:
        logger.info(f"ffound task value is not none. task is {task}")
        processor = glue_processors[task]()
        logger.info("got the processor value")
        if label_list is None:
            logger.info("found that label list is none. going to do processor.get_labels()")
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    else:
        logger.info(f"found task value is None. Error.. task is {task}")

    logger.info("going to map labels")
    label_map = {label: i for i, label in enumerate(label_list)}

    logger.info(f"done with label mapping. labels are {label_map}")

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)


    labels = [label_from_example(example) for example in examples]

    logger.info(f"done with getting labels from example .  going to do batch _encoding")
    logger.info(f"value of tokenixer is {tokenizer}")



    examples_no_stopwords=[]
    if(remove_stop_words==True):

        #giving error on hpc. not able to donwload.
        #stop_words = get_stop_words('english')
        stop_words = []
        #manually adding nltk stop words since it was getting dificcult to download this on the fly on the hpc machine
        nltk_stop_words= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                        'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                        't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
                        're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
                        "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                        'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                        "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        stop_words.extend(nltk_stop_words)
        stop_words.extend(["`","\`","--","``","'","\"","''","‘"," — ","-","_","__","=","."])



        for example_sw in examples:
            text_a_tokens=example_sw.text_a.split(" ")
            text_a_tokens_new=[]
            for each_token in text_a_tokens:
                if not each_token.lower() in stop_words:
                    if not each_token.lower() in string.punctuation:
                        text_a_tokens_new.append(each_token)
            example_sw.text_a=" ".join(text_a_tokens_new)

            text_b_tokens=example_sw.text_b.split(" ")
            text_b_tokens_new=[]
            for each_token in text_b_tokens:
                if not each_token.lower() in stop_words:
                    if not each_token.lower() in string.punctuation:
                        text_b_tokens_new.append(each_token)
            example_sw.text_b=" ".join(text_b_tokens_new)

            examples_no_stopwords.append(example_sw)
        examples=examples_no_stopwords

    assert len(examples)>0
    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )


    features = []

    logger.info(f"done with batch_encoding.  going to get inside enumerate features")
    logger.debug(
        f"value of  batch_encoding are {batch_encoding}..")

    for i in tqdm(range(len(examples)),desc="creating features", total=len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    #for i, example in enumerate(examples[:5]):
        # logger.info("*** Example ***")
        # logger.info("guid: %s" % (example.guid))
        # logger.info("features: %s" % features[i])

    return features

    def _glue_convert_pair_examples_to_features(
        examples1: List[InputExample],
        examples2: List[InputExample],
        tokenizer_lex: PreTrainedTokenizer,
        tokenizer_delex: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
    ):
        if max_length is None:
            max_length = tokenizer_lex.max_len

        if task is not None:
            processor = glue_processors[task]()
            if label_list is None:
                label_list = processor.get_labels()
                logger.info("Using label list %s for task %s" % (label_list, task))
            if output_mode is None:
                output_mode = glue_output_modes[task]
                logger.info("Using output mode %s for task %s" % (output_mode, task))

        label_map = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample) -> Union[int, float, None]:
            if example.label is None:
                return None
            if output_mode == "classification":
                return label_map[example.label]
            elif output_mode == "regression":
                return float(example.label)
            raise KeyError(output_mode)

        labels1 = [label_from_example(example) for example in examples1]
        labels2 = [label_from_example(example) for example in examples2]

        assert labels1==labels2

        batch_encoding_lex = tokenizer_lex.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples1], max_length=max_length, pad_to_max_length=True,
        )
        batch_encoding_delex = tokenizer_delex.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples2], max_length=max_length, pad_to_max_length=True,
        )
        assert len(examples1)==len(examples2)
        features = []
        for i in range(len(examples1)):
            inputs1 = {k: batch_encoding_lex[k][i] for k in batch_encoding_lex}
            inputs2 = {k: batch_encoding_delex[k][i] for k in batch_encoding_delex}

            feature1 = InputFeatures(**inputs1, label=labels1[i])
            feature2 = InputFeatures(**inputs2, label=labels2[i])
            feature=(feature1,feature2)

            #feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        for i, example in enumerate(examples1[:5]):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            #logger.info("features: %s" % features[0][i])

        return features

#created:jan2021.
# this is a generic function which will take any list of data pairs and convert all them and return them back as tuples
# this is useful when using multiple teachers and 3 or 4 datasets neeed to be read in parallel
# Eg:In List[List[InputExample]] List[0] is lexicalized dataset, List[0][0] is the first datapoint ( a claim evidence pair) in lex dataset
def _glue_convert_list_of_example_pairs_to_features(
    all_datasets: List[List[InputExample]],
    tokenizer_lex: PreTrainedTokenizer,
    tokenizer_delex: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):

    total_no_of_datapoints=len(all_datasets[0])


    if max_length is None:
        max_length = tokenizer_lex.max_len

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    list_of_lists_of_labels=[]
    for each_dataset in all_datasets:
        assert len(each_dataset)== total_no_of_datapoints
        labels = [label_from_example(example) for example in each_dataset]
        length_labels = len(labels)
        list_of_lists_of_labels.append(labels)

    assert length_labels > 0

    for each_list in list_of_lists_of_labels:
        assert len(each_list) == length_labels

    all_encoded_datasets=[]

    #encode each claim evidence pair (example) in each dataset using the respective tokenizer provided.
    # Note: assumption: first dataset will be lexicalized and hence only the first dataset, lex, will be tokenized using a lexicalized tokenizer while rest of the datasets
    # will be tokenized using a delexicalized dataset
    batch_encoding_lex = tokenizer_lex.batch_encode_plus(
        [(example.text_a, example.text_b) for example in all_datasets[0]], max_length=max_length, pad_to_max_length=True,
    )

    all_encoded_datasets.append(batch_encoding_lex)

    #encode rest of them using delexicalized tokenizer
    for x in range(1,len(all_datasets)):
        batch_encoding_delex = tokenizer_delex.batch_encode_plus(
        [(example.text_a, example.text_b) for example in all_datasets[x]], max_length=max_length, pad_to_max_length=True,
        )
        all_encoded_datasets.append(batch_encoding_delex)

    assert len(all_encoded_datasets) > 0
    assert len(all_encoded_datasets) == len(all_datasets)


    features = []

    #each data point (a combined sentence of claim and evidence) must have 3 forms now.
    # stitch them all together as a single tuple of (datapoint_lex, datapoint_delex_figers, datapoint_delex_oaner)
    # and also wrap them in the class InputFeatures

    #i is the nth data point and k is the 3 fields (input_ids, token_typeids, attention_mask)


    #todo:replace this with a generic function which automatically finds the total number of various types of datasets and creates feature per data point accordingly
    for i in range(total_no_of_datapoints):
            inputs1 = {k: all_encoded_datasets[0][k][i] for k in all_encoded_datasets[0]}
            inputs2 = {k: all_encoded_datasets[1][k][i] for k in all_encoded_datasets[1]}
            inputs3 = {k: all_encoded_datasets[2][k][i] for k in all_encoded_datasets[2]}
            inputs4 = {k: all_encoded_datasets[3][k][i] for k in all_encoded_datasets[3]}

            feature1 = InputFeatures(**inputs1, label=list_of_lists_of_labels[0][i])
            feature2 = InputFeatures(**inputs2, label=list_of_lists_of_labels[0][i])
            feature3 = InputFeatures(**inputs3, label=list_of_lists_of_labels[0][i])
            feature4 = InputFeatures(**inputs4, label=list_of_lists_of_labels[0][i])
            feature=(feature1,feature2,feature3,feature4)
            features.append(feature)


    # for i in range(total_no_of_datapoints):
    #     list_features=[]
    #     for each_dataset in all_encoded_datasets:
    #         inputs = {k: each_dataset[k][i] for k in each_dataset}
    #         feature = InputFeatures(**inputs, label=list_of_lists_of_labels[0][i])
    #         list_features.append(feature)
    #     tuple(list_features)
    #     features.append(feature)
    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines),desc="creating examples",total=len(lines)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_train_examples_set1(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train1.tsv")), "train")

    def get_train_examples_set2(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2.tsv")), "train")

    def get_train_examples_set3(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train3.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class FeverInDomainProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    def get_train_examples_set1(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train1.tsv")), "train")

    def get_train_examples_set2(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2.tsv")), "train")

    def get_train_examples_set3(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train3.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["disagree", "agree", "nei"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class FeverCrossDomainProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    def get_train_examples_set1(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train1.tsv")), "train")

    def get_train_examples_set2(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train2.tsv")), "train")

    def get_train_examples_set3(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train3.tsv")), "train")

    def get_train_examples_given_dataset_index(self, data_dir,index):
        train_file_name="train"+str(index+1)+".tsv"
        return self._create_examples(self._read_tsv(os.path.join(data_dir, train_file_name)), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    #in multiple teacher land, we will be using each trained model to test on corresponding dataset. So now
    # get data from the corresponding test dataset
    def get_test_examples_given_dataset_index(self, data_dir, index=0):
        """See base class."""
        test_file_name="test"+str(index+1)+".tsv"
        #passing dev instead of test to make it read labels also.
        # this is needed because we are using test partition to load cross domain dev dataset
        return self._create_examples(self._read_tsv(os.path.join(data_dir, test_file_name)), "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        #passing dev instead of test to make it read labels also.
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["disagree", "agree", "discuss","unrelated"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines),desc="creating examples",total=len(lines)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = None if test_mode else line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

print("just before ppicking labels in processors")
glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "feverindomain": 3 ,
    "fevercrossdomain": 4
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "feverindomain": FeverInDomainProcessor,
    "fevercrossdomain": FeverCrossDomainProcessor

}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "feverindomain": "classification",
    "fevercrossdomain": "classification",
}
