

"""
note from mithun @Sat Nov 28 15:25:43 MST 2020:

This file loads a trained model and tests it on test partition.

Steps:
- got to mithun_scripts/run_all.sh and set export TASK_TYPE (i.e the type of trained model and data you are using eg."lex" or "combined")
- uncomment the very last line to load model instead of training (i.e comment #./run_glue.sh and uncomment ./load_model_test.sh)
- go to mithun_scripts/get_fever_fnc_data.sh
- find the run corresponding to TASK_TYPE (e.g lex+fnccross domain)
- change the download path of test partition to download cross domain test partition (during development time we were using cross domain dev partition)
- in this file (./load_trained_model_print_attention_weights.py) change config_file_touse to point to right config file
e.g:( CONFIG_FILE_TO_TEST_LEX_MODEL_WITH_HPC or CONFIG_FILE_TO_TEST_LEX_MODEL_WITH_LAPTOP)
- To run: go to huggingface/mithun_scripts command line sand use the run script (e.g.using: bash run_all.sh --epochs_to_run 2 --machine_to_run_on laptop --use_toy_data true --download_fresh_data true)
Note: The download_fresh_data true has to be done only once per  TASK_TYPE
Note: if you are running from pycharm, the configuration you should use is "load combined trained model and get attention weights"



additional steps if running on hpc:


- find the name of the folder on hpc
- Go to local machine. 
- change it in 3 files
1)run_all.sh (2instances)
2)run_on_hpc_ocelote_venv_array.sh (3 instances)
3) the corresponding config file you are going to use (see below) 2 instances


"""



""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
CONFIG_FILE_TO_TEST_LEX_MODEL_WITH_LAPTOP= "config_for_attention_visualization_for_loading_lex_model_laptop.py"
CONFIG_FILE_TO_TEST_LEX_MODEL_WITH_HPC= "config_for_attention_visualization_for_loading_lex_model_hpc.py"
CONFIG_FILE_TO_TEST_STUTEACHER_MODEL_WITH_LAPTOP="config_for_attention_visualization_for_loading_stuteacher_model_laptop.py"
CONFIG_FILE_TO_TEST_STUTEACHER_MODEL_WITH_HPC="config_for_attention_visualization_for_loading_stuteacher_model_hpc.py"
config_file_touse = CONFIG_FILE_TO_TEST_STUTEACHER_MODEL_WITH_HPC
NO_OF_LAYERS=12
NO_OF_HEADS_PER_LAYER=12


import csv
import configparser
import sys, getopt
import re
import shutil
import warnings
from pathlib import Path
from typing import  Callable, Dict, List, Optional, Tuple
from torch.nn import CrossEntropyLoss, MSELoss
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
from stop_words import get_stop_words
import string
import spacy
nlp = spacy.load("en_core_web_sm")

#os.chdir("/content/gdrive/My Drive/colab_fall2020/transformers2/transformers/src")
from transformers.data.data_collator import DataCollator, default_data_collator, collate_batch_parallel_datasets
from transformers.file_utils import is_apex_available, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    PredictionOutput,
    is_wandb_available,
)

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)
#os.chdir("/content/gdrive/My Drive/colab_fall2020/transformers2/transformers/examples/text-classification")


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import git
import numpy as np


import wget
import torch


if is_apex_available():
    from apex import amp


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False

def is_tensorboard_available():
    return _has_tensorboard


if is_wandb_available():
    import wandb


logger = logging.getLogger(__name__)

class StudentTeacherTrainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,

    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel`):
            The model to train, evaluate or use for predictions.
        args (:class:`~transformers.TrainingArguments`):
            The arguments to tweak training.
        data_collator (:obj:`DataCollator`, `optional`, defaults to :func:`~transformers.default_data_collator`):
            The function to use to from a batch from a list of elements of :obj:`train_dataset` or
            :obj:`dev_dataset`.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for evaluation.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        prediction_loss_only (:obj:`bool`, `optional`, defaults to `False`):
            When performing evaluation and predictions, only returns the loss.
        tb_writer (:obj:`SummaryWriter`, `optional`):
            Object to write to TensorBoard.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.

    """

    models: {}
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    test_dataset = Optional[Dataset]
    test_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    eval_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
            self,
            tokenizer_delex,
            tokenizer_lex,
            models: {},
            args: TrainingArguments,
            train_datasets: {},
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            test_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            eval_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            data_collator: Optional[DataCollator] = None,
            prediction_loss_only=False,
            tb_writer: Optional["SummaryWriter"] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None)
    ):

        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.
        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.lex_teacher_model = models.get("teacher").to(args.device)
        self.delex_student_model = models.get("student").to(args.device)

        self.eval_dataset = eval_dataset
        self.lex_tokenizer = tokenizer_lex
        self.delex_tokenizer = tokenizer_delex

        ###for fnc score evaluation
        self.test_dataset = test_dataset
        self.test_compute_metrics = test_compute_metrics
        self.eval_compute_metrics = eval_compute_metrics
        self.compute_metrics = None
        # even though we train two models using student teacher architecture we weill only use the student model to do evaluation on fnc-dev delex dataset
        self.model = None
        self.args = args
        self.default_data_collator = default_data_collator
        self.data_collator = collate_batch_parallel_datasets
        self.train_dataset_combined = train_datasets.get("combined")
        self.eval_dataset = eval_dataset
        self.compute_metrics = None
        self.prediction_loss_only = prediction_loss_only
        self.optimizer, self.lr_scheduler = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                        "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                        + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

    def write_predictions_to_disk(self, plain_text, gold_labels, predictions_logits, file_to_write_predictions,
                                  test_dataset):
        predictions_argmaxes = np.argmax(predictions_logits, axis=1)
        sf = torch.nn.Softmax(dim=1)
        predictions_softmax = sf(torch.FloatTensor(predictions_logits))
        if self.is_world_master():
            with open(file_to_write_predictions, "w") as writer:
                logger.info(f"***** (Going to write Test results to disk at {file_to_write_predictions} *****")
                writer.write("index\t gold\tprediction_logits\t prediction_label\tplain_text\n")
                for index, (gold, pred_sf, pred_argmax, plain) in enumerate(
                        zip(gold_labels, predictions_softmax, predictions_argmaxes, plain_text)):
                    gold_string = test_dataset.get_labels()[gold]
                    pred_string = test_dataset.get_labels()[pred_argmax]
                    writer.write(
                        "%d\t%s\t%s\t%s\t%s\n" % (index, gold_string, str(pred_sf.tolist()), pred_string, plain))

    def predict_given_trained_model(self, model, test_dataset):
        predictions_argmaxes = self.predict(test_dataset, model).predictions_argmaxes

    def predict(self, test_dataset: Dataset, model_to_test_with) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on.
        Returns:
            `NamedTuple`:
            predictions (:obj:`np.ndarray`):
                The predictions on :obj:`test_dataset`.
            label_ids (:obj:`np.ndarray`, `optional`):
                The labels (if the dataset contained some).
            metrics (:obj:`Dict[str, float]`, `optional`):
                The potential dictionary of metrics (if the dataset contained labels).
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self.prediction_loop(test_dataloader, model_to_test_with, description="Prediction")

    def evaluate(self, model_to_test_with, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`eval_compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output, plain_text = self.prediction_loop(eval_dataloader, model_to_test_with, description="Evaluation")
        gold_labels = output.label_ids
        predictions = output.predictions

        self.log(output.metrics)
        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics, plain_text, gold_labels, predictions

    def _prepare_inputs(
            self, inputs: Dict[str, torch.Tensor], model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        # Our model outputs do not work with DataParallel, so forcing return tuple.
        if isinstance(model, nn.DataParallel):
            inputs["return_tuple"] = True
        return inputs

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset_combined is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset_combined)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset_combined)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset_combined)
            )
        data_loader = DataLoader(
            self.train_dataset_combined,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )
        return data_loader

    def get_train_dataloader_two_parallel_datasets(self) -> DataLoader:
        if self.train_dataset_combined is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset_combined)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset_combined)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset_combined)
            )
        data_loader = DataLoader(
            self.train_dataset_combined,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.default_data_collator
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_torch_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.default_data_collator,
        )

        return data_loader

    def get_optimizers_for_student_teacher(
            self, num_training_steps: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        update: will combine parameters
        """
        if self.optimizer is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.lex_teacher_model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.lex_teacher_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.delex_student_model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.delex_student_model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },

        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def get_optimizer(
            self, model, num_training_steps: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        update: will combine parameters
        """
        if self.optimizer is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.
        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:
        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.lex_teacher_model, log=os.getenv("WANDB_WATCH", "gradients"),
                log_freq=max(100, self.args.logging_steps)
            )
            wandb.watch(
                self.delex_student_model, log=os.getenv("WANDB_WATCH", "gradients"),
                log_freq=max(100, self.args.logging_steps)
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def get_plaintext_given_dataloader(self):
        pass
        # return eval_dataloader.dataset.features

    def evaluate_on_test_partition(self, model_to_test_with, test_dataset: Optional[Dataset] = None, ) -> Dict[
        str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`eval_compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            test_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`.
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        eval_dataloader = self.get_test_dataloader(test_dataset)

        output, plain_text = self.prediction_loop(eval_dataloader, model_to_test_with, description="Evaluation")
        gold_labels = output.label_ids
        predictions = output.predictions

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        return output.metrics, plain_text, gold_labels, predictions

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        # if hasattr(self, "_log"):
        #     warnings.warn(
        #         "The `_log` method is deprecated and won't be called in a future version, define `log` in your subclass.",
        #         FutureWarning,
        #     )
        #     return self._log(logs, iterator=iterator)

        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        '"%s" of type %s for key "%s" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_master():
                wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(output)

    def _intermediate_eval(self, datasets, epoch, output_eval_file, description, model_to_test_with):

        """
        Helper function to call eval() method if and when you want to evaluate after say each epoch,
        instead having to wait till the end of all epochs
        Returns:
        """

        logger.info(f"*******End of training.Going to run evaluation on {description} ")

        if "dev" in description:
            self.compute_metrics = self.eval_compute_metrics
        else:
            if "test" in description:
                self.compute_metrics = self.test_compute_metrics

        assert self.compute_metrics is not None
        # Evaluation
        eval_results = {}
        datasetss = [datasets]
        for dataset in datasetss:
            eval_result = None
            if "dev" in description:
                eval_result, plain_text, gold_labels, predictions = self.evaluate(model_to_test_with,
                                                                                  eval_dataset=dataset)
            else:
                if "test" in description:
                    eval_result, plain_text, gold_labels, predictions = self.evaluate_on_test_partition(
                        model_to_test_with, test_dataset=dataset)
            assert eval_result is not None

            if self.is_world_master():
                with open(output_eval_file, "a") as writer:
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)
        return eval_result, plain_text, gold_labels, predictions

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        assert self.model is not None
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor,], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

        inputs = self._prepare_inputs(inputs, model)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss, logits = outputs[:2]
                loss = loss.mean().item()
            else:
                loss = None
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, logits.detach(), labels)

    def get_git_info(self):
        repo = git.Repo(search_parent_directories=True)

        repo_sha = str(repo.head.object.hexsha),
        repo_short_sha = str(repo.git.rev_parse(repo_sha, short=6))

        repo_infos = {
            "repo_id": str(repo),
            "repo_sha": str(repo.head.object.hexsha),
            "repo_branch": str(repo.active_branch),
            "repo_short_sha": repo_short_sha
        }
        return repo_infos

    def train_1teacher_1student(self, model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader_two_parallel_datasets()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                    self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        model_teacher = self.lex_teacher_model
        model_student = self.delex_student_model

        weight_consistency_loss = 1
        weight_classification_loss = 0.0875

        optimizer = None
        scheduler = None

        # these flags are used for testing purposes. IDeally when running in student teacher mode this should be
        # flag_run_both=True. Other two flags are to test by loading each of these models independently from within
        # the same trainer class
        flag_run_teacher_alone = False
        flag_run_student_alone = True
        flag_run_both = False

        if flag_run_both:
            optimizer, scheduler = self.get_optimizers_for_student_teacher(num_training_steps=self.args.lr_max_value)
            assert optimizer is not None
            assert scheduler is not None
        else:
            if flag_run_teacher_alone:
                optimizer, scheduler = self.get_optimizer(model_teacher, num_training_steps=self.args.lr_max_value)
                assert optimizer is not None
                assert scheduler is not None
            else:
                if flag_run_student_alone:
                    optimizer, scheduler = self.get_optimizer(model_student, num_training_steps=self.args.lr_max_value)
                    assert optimizer is not None
                    assert scheduler is not None

        # Check if saved optimizer or scheduler states exist
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model_teacher, optimizer = amp.initialize(model_teacher, optimizer, opt_level=self.args.fp16_opt_level)
            model_student, optimizer = amp.initialize(model_student, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()

            model_teacher = torch.nn.DataParallel(model_teacher)
            model_student = torch.nn.DataParallel(model_student)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model_teacher = torch.nn.parallel.DistributedDataParallel(
                model_teacher,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )
        if self.args.local_rank != -1:
            model_student = torch.nn.parallel.DistributedDataParallel(
                model_student,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                    self.args.train_batch_size
                    * self.args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                        len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss_lex_float = 0.0
        tr_loss_delex_float = 0.0
        logging_loss = 0.0
        model_teacher.zero_grad()
        model_student.zero_grad()

        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        git_details = self.get_git_info()

        # empty out the file which stores intermediate evaluations
        output_dir_absolute_path = os.path.join(os.getcwd(), self.args.output_dir)
        dev_partition_evaluation_output_file_path = output_dir_absolute_path + "intermediate_evaluation_on_dev_partition_results_" + \
                                                    git_details['repo_short_sha'] + ".txt"
        # empty out the
        with open(dev_partition_evaluation_output_file_path, "w") as writer:
            writer.write("")

        test_partition_evaluation_output_file_path = output_dir_absolute_path + "intermediate_evaluation_on_test_partition_results_" + \
                                                     git_details['repo_short_sha'] + ".txt"
        # empty out the
        with open(test_partition_evaluation_output_file_path, "w") as writer:
            writer.write("")

        # empty out the file which stores intermediate evaluations
        predictions_on_test_file_path = output_dir_absolute_path + "predictions_on_test_partition_" + git_details[
            'repo_short_sha'] + ".txt"
        with open(predictions_on_test_file_path, "w") as writer:
            writer.write("")

        best_fnc_score = 0
        best_acc = 0

        # for each epoch

        for epoch in train_iterator:
            logger.debug("just got inside for epoch in train_iterator")

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                logger.debug("found that is_torch_tpu_available is true")

                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="batches", disable=not self.is_local_master())
            else:
                logger.debug("found that is_torch_tpu_available is false")
                epoch_iterator = tqdm(train_dataloader, desc="batches", disable=not self.is_local_master())

            # for each batch
            for step, (input_lex, input_delex) in enumerate(epoch_iterator):
                logger.debug("just got inside for step in enumerate epoch_iterator. i.e for each batch")

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                assert input_lex['labels'].tolist() == input_delex['labels'].tolist()

                # model returns # (loss), logits, (hidden_states), (attentions)
                tr_loss_lex, outputs_lex = self.get_classification_loss(model_teacher, input_lex, optimizer)
                tr_loss_delex, outputs_delex = self.get_classification_loss(model_student, input_delex, optimizer)

                if (flag_run_both):
                    combined_classification_loss = tr_loss_lex + tr_loss_delex

                else:
                    if (flag_run_teacher_alone):
                        combined_classification_loss = tr_loss_lex
                    else:
                        if (flag_run_student_alone):
                            combined_classification_loss = tr_loss_delex

                logger.debug("finished getting classification loss")

                # outputs contains in that order # (loss), logits, (hidden_states), (attentions)-src/transformers/modeling_bert.py
                logits_lex = outputs_lex[1]
                logits_delex = outputs_delex[1]
                consistency_loss = self.get_consistency_loss(logits_lex, logits_delex, "mse")

                if (flag_run_both):
                    combined_loss = (weight_classification_loss * combined_classification_loss) + (
                                weight_consistency_loss * consistency_loss)
                else:
                    combined_loss = combined_classification_loss

                if self.args.fp16:
                    logger.info("self.args.fp16 is true")
                    with amp.scale_loss(combined_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    with amp.scale_loss(combined_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    logger.debug("self.args.fp16 is false")
                    combined_loss.backward()

                    logger.debug("just got done with combined_loss.backward()")

                tr_loss_lex_float += tr_loss_lex.item()
                tr_loss_delex_float += tr_loss_delex.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    logger.debug("got inside if condition for if(step+1. i.e last step in epoch)")

                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        if (flag_run_both):
                            torch.nn.utils.clip_grad_norm_(model_student.parameters(), self.args.max_grad_norm)
                            torch.nn.utils.clip_grad_norm_(model_teacher.parameters(), self.args.max_grad_norm)

                        else:
                            if (flag_run_teacher_alone):
                                torch.nn.utils.clip_grad_norm_(model_teacher.parameters(), self.args.max_grad_norm)
                            else:
                                if (flag_run_student_alone):
                                    torch.nn.utils.clip_grad_norm_(model_student.parameters(), self.args.max_grad_norm)

                    logger.debug("just done with grad clipping)")

                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()
                        logger.debug("just done withn optimixer.step)")
                    scheduler.step()
                    if (flag_run_both):
                        model_teacher.zero_grad()
                        model_student.zero_grad()

                    else:
                        if (flag_run_teacher_alone):
                            model_teacher.zero_grad()
                        else:
                            if (flag_run_student_alone):
                                model_student.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                            self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        logs["loss"] = (tr_loss_lex_float - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss_lex_float

                        self.log(logs)

                        # if self.args.evaluate_during_training:
                        #     self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        # update: in student teacher setting since there are way too many model words going on, we will ezxplicitly pass the model to save

                        if (flag_run_both):
                            if hasattr(model_teacher, "module"):
                                assert model_teacher.module is self.lex_teacher_model.module
                                assert model_student.module is self.delex_student_model.module

                            else:
                                assert model_teacher is self.lex_teacher_model
                                assert model_student is self.delex_student_model

                            self.model = model_teacher
                            output_dir = os.path.join(self.args.output_dir,
                                                      f"model_teacher_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                            assert self.model is not None
                            self.save_model(output_dir)

                            self.model = model_student
                            output_dir = os.path.join(self.args.output_dir,
                                                      f"model_student_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                            assert self.model is not None
                            self.save_model(output_dir)


                        else:
                            if (flag_run_teacher_alone):
                                if hasattr(model_teacher, "module"):
                                    assert model_teacher.module is self.lex_teacher_model.module
                                else:
                                    assert model_teacher is self.lex_teacher_model
                                self.model = model_teacher
                                output_dir = os.path.join(self.args.output_dir,
                                                          f"model_teacher_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                                self.save_model(output_dir)
                            else:
                                if (flag_run_student_alone):
                                    if hasattr(model_teacher, "module"):
                                        assert model_student.module is self.delex_student_model.module
                                    else:
                                        assert model_student is self.delex_student_model
                                self.model = model_student
                                output_dir = os.path.join(self.args.output_dir,
                                                          f"model_student_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                                self.save_model(output_dir)

                        # Save model checkpoint

                        if hasattr(model_student, "module"):
                            assert model_student.module is self.delex_student_model
                        else:
                            assert model_student is self.delex_student_model
                        # Save model checkpoint

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_torch_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

            trained_model = None
            if (flag_run_both):
                trained_model = model_student
            else:
                if (flag_run_teacher_alone):
                    trained_model = model_teacher
                else:
                    if (flag_run_student_alone):
                        trained_model = model_student

            assert trained_model is not None

            dev_partition_evaluation_result, plain_text, gold_labels, predictions = self._intermediate_eval(
                datasets=self.eval_dataset,
                epoch=epoch,
                output_eval_file=dev_partition_evaluation_output_file_path,
                description="dev_partition", model_to_test_with=trained_model)

            test_partition_evaluation_result, plain_text, gold_labels, predictions_logits = self._intermediate_eval(
                datasets=self.test_dataset,
                epoch=epoch, output_eval_file=test_partition_evaluation_output_file_path, description="test_partition",
                model_to_test_with=trained_model)

            fnc_score_test_partition = test_partition_evaluation_result['eval_acc']['cross_domain_fnc_score']
            accuracy_test_partition = test_partition_evaluation_result['eval_acc']['cross_domain_acc']

            if fnc_score_test_partition > best_fnc_score:
                best_fnc_score = fnc_score_test_partition

                logger.info(f"found that the current fncscore:{fnc_score_test_partition} in epoch "
                            f"{epoch} beats the bestfncscore so far i.e ={best_fnc_score}. going to prediction"
                            f"on test partition and save that and model to disk")
                # if the accuracy or fnc_score_test_partition beats the highest so far, write predictions to disk

                self.write_predictions_to_disk(plain_text, gold_labels, predictions_logits,
                                               predictions_on_test_file_path,
                                               self.test_dataset)

                # Save model checkpoint
                self.model = trained_model
                output_dir = os.path.join(self.args.output_dir)
                self.save_model(output_dir)

                # if self.is_world_master():
                #     self._rotate_checkpoints()

                if is_torch_tpu_available():
                    xm.rendezvous("saving_optimizer_states")
                    xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                elif self.is_world_master():
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if accuracy_test_partition > best_acc:
                best_acc = accuracy_test_partition

            logger.info(
                f"********************************end of epoch {epoch+1}************************************************************************")
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        # Note: returning for testing purposes only. All performance evaluation measures by now are written to disk.
        # Note that the assumption here is that the test will be run for 1 epoch only. ELse have to return the best dev and test partition scores
        return dev_partition_evaluation_result, test_partition_evaluation_result

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        # if hasattr(self, "_log"):
        #     warnings.warn(
        #         "The `_log` method is deprecated and won't be called in a future version, define `log` in your subclass.",
        #         FutureWarning,
        #     )
        #     return self._log(logs, iterator=iterator)

        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        log_for_wandb = {}
        if self.tb_writer:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, self.global_step)
                    log_for_wandb[k] = v
                else:
                    if isinstance(v, (dict)):
                        for k2, v2 in v.items():
                            if isinstance(v2, (int, float)):
                                self.tb_writer.add_scalar(k2, v2, self.global_step)
                                log_for_wandb[k2] = v2
                            else:
                                logger.warning(
                                    f"Trainer is attempting to log a valuefor key {k2}as a scalar."
                                    f"This invocation of Tensorboard's writer.add_scalar()"
                                    f" is incorrect so we dropped this attribute.",
                                )
                                logger.debug(v2)
            self.tb_writer.flush()
        if is_wandb_available():
            if self.is_world_master():
                if len(log_for_wandb.items()) > 0:
                    wandb.log(log_for_wandb, step=int(self.epoch))
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.debug(output)

    def _training_step(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def get_classification_loss(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        '''
        similar to _training_step however returns loss instead of doing .backward
        Args:
            model:
            inputs:
            optimizer:
        Returns:loss
        '''
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss, outputs

    def get_logits(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ):
        '''
        similar to _training_step however returns loss instead of doing .backward
        Args:
            model:
            inputs:
            optimizer:
        Returns:loss
        '''
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        return outputs

    def get_consistency_loss(
            self, logit1, logit2, loss_function):
        '''
        similar to _training_step however returns loss instead of doing .backward
        Args:

            model:
            inputs:
            optimizer:
        Returns:loss
        '''

        if (loss_function == "mse"):
            loss_fct = MSELoss()
            loss = loss_fct(logit1.view(-1), logit2.view(-1))

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss

    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def prediction_loop(
            self, dataloader: DataLoader, model_to_test_with, description: str,
            prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = model_to_test_with
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.debug("***** Running %s at epoch number:%s *****", description, self.epoch)
        logger.debug("  Num examples = %d", self.num_examples(dataloader))
        logger.debug("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None
        plain_text_full = []
        for inputs in tqdm(dataloader, desc=description):
            plain_text_batch = self.delex_tokenizer.batch_decode(inputs['input_ids'])
            plain_text_full.extend(plain_text_batch)
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)
            if loss is not None:
                eval_losses.append(loss)
            if logits is not None:
                preds = logits if preds is None else torch.cat((preds, logits), dim=0)
            if labels is not None:
                label_ids = labels if label_ids is None else torch.cat((label_ids, labels), dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")
        logger.debug(f" value of local rank is {self.args.local_rank}")
        if self.args.local_rank != -1:
            logger.info(f"found that local_rank is not minus one. value of local rank is {self.args.local_rank}")
            import sys
            sys.exit(1)
            # In distributed mode, concatenate all results from all nodes:
            if preds is not None:
                preds = self.distributed_concat(preds, num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))
        elif is_torch_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if preds is not None:
                preds = xm.mesh_reduce("eval_preds", preds, torch.cat)
            if label_ids is not None:
                label_ids = xm.mesh_reduce("eval_label_ids", label_ids, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if preds is not None:
            preds = preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            if self.compute_metrics is None:
                logger.error("compute_metrics  is none. going to exit")
            if preds is None:
                logger.error("preds  is none. going to exit")
            if label_ids is None:
                logger.error("label_ids  is none. going to exit")
            import sys
            sys.exit(1)
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        assert plain_text_full is not None
        assert len(plain_text_full) == len(dataloader.dataset.features)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics), plain_text_full



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
def read_and_merge_config_entries(base_file_path,machine_to_run_on):



    assert len(config_file_touse)>0
    config = configparser.ConfigParser()

    config.read(base_file_path+config_file_touse)
    #assert not len(config.sections())==0
    combined_configs=[]

    for each_section in config.sections():
        for (each_key, each_val) in config.items(each_section):
            #some config entries of type bool just need to exist. doesnt need x=True.
            # so now have to strip True out, until we findc a way to be able to pass it as bool itself
            # so if True is a value, append only key
            if (each_val=="True"):
                combined_configs.append("--"+each_key)
            else:
                combined_configs.append("--" + each_key)
                combined_configs.append(str(each_val).replace("\"",""))
    combined_configs_str=" ".join(combined_configs)

    return combined_configs_str


  
def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


    
def run_loading_and_testing(model_args, data_args, training_args):
    # Setup logging
    git_details=get_git_info()

    log_file_name=git_details['repo_short_sha']+"_"+(training_args.task_type)+"_"+(training_args.subtask_type)+"_"+str(model_args.model_name_or_path).replace("-","_")+"_"+data_args.task_name+".log"
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

    # Load pretrained model_teacher and tokenizer_lex
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
    tokenizer_lex = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        force_download=True,
        do_lower_case=False
    )

    #when in student-teacher mode, you need two tokenizers, one for lexicalized data, and one for the delexicalized data
    # the regular tokenizer_lex will be used for lexicalized data and special one for delexicalized
    tokenizer_delex = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        force_download=True,
        tokenizer_type="delex",
        do_lower_case=False
    )


    #this is needed for visualization
    config.output_attentions=True


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
        model_for_bert = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    if (training_args.do_train_1student_1teacher == True):
        # the task type must be delex, . also make sure the corresponding data has been downloaded in get_fever_fnc_data.sh
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

    if (training_args.do_train_1student_1teacher == True):
        test_dataset = (
            GlueDataset(data_args, tokenizer=tokenizer_delex, task_type="delex", mode="test",
                        cache_dir=model_args.cache_dir)
            if training_args.do_predict
            else None
        )
    else:
        if (training_args.task_type == "lex"):
            test_dataset = (
                GlueDataset(data_args, tokenizer=tokenizer_lex,task_type="lex", mode="test",
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

    def visualize(tokenizer,model):
        ###code for visualization from  https://github.com/jessevig/bertviz
        sentence_a = "The cat sat on the mat"
        sentence_b = "The dog lay on the rug"
        inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
        token_type_ids = inputs['token_type_ids']
        input_ids = inputs['input_ids']
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        input_id_list = input_ids[0].tolist()  # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)



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

    def sort_weights(dict_layer_head_weights):
        return {k: v for k, v in sorted(dict_layer_head_weights.items(), key=lambda item: item[1],reverse=True)}



    def get_cross_claim_evidence_percentage_attention_weights(dataloader, model, tokenizer):
        attention = claim_evidence_plain_text = None
        logger.info(f"getting into fn get_cross_claim_evidence_percentage_attention_weights ")
        # create a dictionary to store overall attention of a given head and a layer. maybe can eventually store it in a matrix
        # lets start with 12th layer, 12th attention head- eventually we wil need to create a 12x12 matrix of such dicts for 12 layers and 12 heads
        dict_tokens_attention = {}

        cross_fit = "cross_sentence.csv"

        write_to_csv_file(cross_fit, 0, 0, 0)
        for layer in range(11, NO_OF_LAYERS):
            logger.info(f"getting into layer number:{layer}")
            print(f"getting into layer number:{layer}")
            for head in range(11, NO_OF_HEADS_PER_LAYER):
                logger.info(f"getting into head number:{head}")
                print(f"getting into head number:{head}")
                dict_unique_tokens_attention_weights = {}
                data_counter = 1
                total_length_datapoints = len(dataloader)
                # go through the entire dataset (usually test or dev partition), and for each claim evidence pair,
                # run it through the trained model, which then predicts the label, along with attention it places on each token.

                for_all_datapoints_total_attention_weight_that_came_from_cross_datapoints = 0
                for_all_datapoints_total_attention_weight_that_came_from_within_itself = 0
                data_counter = 1


                for each_claim_evidence_pair in dataloader:
                    print(f"data point:{data_counter}/{total_length_datapoints} ")
                    logger.info(f"data point:{data_counter}/{total_length_datapoints} ")
                    data_counter = data_counter + 1

                    token_type_ids = each_claim_evidence_pair.token_type_ids
                    input_ids = each_claim_evidence_pair.input_ids

                    assert len(token_type_ids) == len(input_ids)
                    input_ids_tensor = None
                    token_type_ids_tensor = None
                    if (training_args.machine_to_run_on == "hpc") and torch.cuda.is_available():
                        input_ids_tensor = torch.cuda.LongTensor(np.reshape(input_ids, (1, len(input_ids))))
                        token_type_ids_tensor = torch.cuda.LongTensor(
                            np.reshape(token_type_ids, (1, len(token_type_ids))))
                    if (training_args.machine_to_run_on == "laptop"):
                        input_ids_tensor = torch.LongTensor(np.reshape(input_ids, (1, len(input_ids))))
                        token_type_ids_tensor = torch.LongTensor(np.reshape(token_type_ids, (1, len(token_type_ids))))

                    assert input_ids_tensor is not None
                    assert token_type_ids_tensor is not None
                    attention = model(input_ids_tensor, token_type_ids=token_type_ids_tensor)[-1]
                    tokens = tokenizer.decode_return_list(input_ids, clean_up_tokenization_spaces=True)

                    try:
                        assert len(tokens) == len(input_ids)
                    except AssertionError:
                        print(f"len(tokens) == {len(tokens)}")
                        print(f"len(input_ids) == {len(input_ids)}")
                        print(f"(tokens) == {(tokens)}")
                        print(f"(input_ids) == {(input_ids)}")
                        print("assertion error exiting")
                        exit()

                        # For each data point (i.e one claim-evidence pair) get the attention given to each of the
                        # tokens store it in a dictionary and do it for all data points

                    for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence, \
                    for_this_claim_what_is_the_total_attention_weight_that_came_from_other_tokens_in_claim = find_attention_percentage_across_claim_ev_both_directions(attention, tokens, layer, head)







                    for_all_datapoints_total_attention_weight_that_came_from_cross_datapoints += for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence
                    for_all_datapoints_total_attention_weight_that_came_from_within_itself += for_this_claim_what_is_the_total_attention_weight_that_came_from_other_tokens_in_claim

                # so at the end of all data points,calculat overall for all claims what percentage attention came from claims and evidence

                logger.info(
                    f"for_all_datapoints_total_attention_weight_that_came_from_cross_datapoints= {for_all_datapoints_total_attention_weight_that_came_from_cross_datapoints} ")

                logger.info(
                    f"for_all_datapoints_total_attention_weight_that_came_from_within_itself= {for_all_datapoints_total_attention_weight_that_came_from_within_itself} ")


                percent_from_evidence = for_all_datapoints_total_attention_weight_that_came_from_cross_datapoints * 100 / (
                            for_all_datapoints_total_attention_weight_that_came_from_cross_datapoints + for_all_datapoints_total_attention_weight_that_came_from_within_itself)
                logger.info(f"for layer:{layer} head {head} percent_attn_from_all_cross_sentence_tokens={percent_from_evidence}")
                append_to_csv_file(cross_fit,layer,head,percent_from_evidence)
                    #append_to_csv_file(output_file_name,layer, head, percentage):



    # out of all the tokens find what percentage of attention goes to NER tags
    def find_percentage_attention_given_to_figer_entities(dict_layer12_head_12,figer_set):
        total_attention_on_all_tokens = 0
        attention_on_ner_tokens = 0
        for token,weight in dict_layer12_head_12.items():
            total_attention_on_all_tokens = total_attention_on_all_tokens + weight
            if token in figer_set:
                attention_on_ner_tokens=attention_on_ner_tokens+weight
        figer_percent_attention=attention_on_ner_tokens*100/total_attention_on_all_tokens
        logger.info(f"figer_percent_attention={figer_percent_attention}")
        return figer_percent_attention

    def find_percentage_attention_given_to_ner_entities(dict_layer12_head_12,test_dataset):
        total_attention_on_all_tokens = 0
        attention_on_ner_tokens = 0
        for token,weight in dict_layer12_head_12.items():
            total_attention_on_all_tokens = total_attention_on_all_tokens + weight
            is_ner = find_ner_or_not(token,test_dataset)
            if is_ner:
                attention_on_ner_tokens=attention_on_ner_tokens+weight
        ner_percent_attention=attention_on_ner_tokens*100/total_attention_on_all_tokens
        logger.info(f"ner_percent_attention={ner_percent_attention}")
        return ner_percent_attention


    def remove_stop_words_punctuations_etc(dict_layer12_head_12):

        stop_words = get_stop_words('english')

        # manually adding nltk stop words since it was getting dificcult to download this on the fly on the hpc machine
        nltk_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                           "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                           'himself',
                           'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
                           'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                           "that'll",
                           'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                           'had',
                           'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                           'because',
                           'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                           'into',
                           'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                           'in',
                           'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                           'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                           'other',
                           'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                           's',
                           't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
                           'o',
                           're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                           'doesn',
                           "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                           'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
                           'shouldn',
                           "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
        stop_words.extend(nltk_stop_words)
        stop_words.extend(["`", "\`", "--", "``", "'", "\"", "''", "‘", " — ", "-", "_", "__", "=", "."])
        stop_words.extend(["[CLS]", "[SEP]","[PAD]"])

        new_dict1 = {key: val for key, val in dict_layer12_head_12.items() if key not in stop_words}
        new_dict2 = {key: val for key, val in new_dict1.items() if key.lower() not in stop_words}
        new_dict3 = {key: val for key, val in new_dict2.items() if key.lower() not in string.punctuation}

        return new_dict3




    def find_ner_or_not(token,test_dataset):
        '''
        GO through the dictionary of all NER tags (that were created while the dataset was read)
        and check if the given token is an NER entity or not
        '''
        assert test_dataset.ner_tags is not None
        if not (test_dataset.ner_tags):
            print("ner_tags is null going to exit")
            import sys
            sys.exit(1)

        if(token in test_dataset.ner_tags):
            return True
        else:
            return False

    def get_figer_tags():
        if (training_args.machine_to_run_on == "laptop"):
            f = open("./figer_tags.txt", "r")
        else:
            f = open("figer_tags.txt", "r")
        all_tags = []
        for x in f:
            split_tab = x.split("\t")
            for y in split_tab:
                z = y.split("/")
                for a in z:
                    all_tags.append(a.strip())
        return (set(all_tags))


    def write_to_file(output_file_name,dict_layer_head):
        # write the aggregated sorted attention weights to disk
        with open(output_file_name, "w") as writer:
            logger.info(f"***** (Going to write attention results to disk at {output_file_name} *****")
            print(f"***** (Going to write attention results to disk at {output_file_name} *****")
            for k, v in dict_layer_head.items():
                writer.write(f"{k}:{v}")
                writer.write(f"\n")

    def append_to_file(output_file_name,dict_layer_head):
        # write the aggregated sorted attention weights to disk
        with open(output_file_name, "a+") as writer:
            logger.info(f"***** (Going to write attention results to disk at {output_file_name} *****")
            for k, v in dict_layer_head.items():
                writer.write(f"{k}:{v}")
                writer.write(f"\n")

    def append_to_csv_file(output_file_name,layer, head, percentage):
        # write the aggregated sorted attention weights to disk
        with open(output_file_name, "a+") as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            logger.info(f"***** (Going to write csv file to disk at {output_file_name} . this is for layer:{layer} head:{head}*****")
            print(
                f"***** (Going to write csv file to disk at {output_file_name} . this is for layer:{layer} head:{head}*****")
            writer.writerow([layer,head,percentage])

    def write_to_csv_file(output_file_name, layer, head, percentage):
            # write the aggregated sorted attention weights to disk
            with open(output_file_name, "w") as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                logger.info(
                    f"***** (Going to write csv file to disk at {output_file_name} . this is for layer:{layer} head:{head}*****")
                print(
                    f"***** (Going to write csv file to disk at {output_file_name} . this is for layer:{layer} head:{head}*****")
                writer.writerow([layer, head, percentage])

    """
    Aim: find what percentage of attention is given to intra-sentence (in this case claim to claim or evidence to evidence)
    as opposed to cross-sentence (i.e while in claim how much attention came from words in evidence and vice versa)
    
    steps:
    - Go through the claim-evidence pair and look for where the first [SEP] tag comes in. That should be where claim ends 
    and evidence starts.  say the length of claim evidence pair if 128 tokens and the claim ends after say token 30.
    
    
    - for each token in claim:
        - for each token, there will be attention coming from 127 other tokens and itself. 
        - find total attention weight coming from tokens before 30 
        - find total attention weight coming from tokens after 30.
    -sum up before 30 and after 30 for each of the n tokens in claim
    - so at the end you should be able to say for entire claim, 35% of attention came from claim itself and 65% came from evidence
    - get the total number. for all claims overall, 25% of attention came from claims itself and 75% from its evidence.
    - now compare these numbers between teacher and student. (i.e using model trained on lex and model trained using 
    student teacher architecture on delexicalized data)
        
    
    - do this for one head. preferably the last one..layer 11, head 11
    - do first from claim to evidence.
    - then do both directions..i.e for evidence, claim will be cross sentence etc
    """

    def find_attention_percentage_across_claim_ev(attention, tokens, layer, head):

        # first go through the entire tokens and find where first [SEP] is. that is where the
        # claim ends and evidence starts
        separator_token = 0
        for index, token in enumerate(tokens):
            if (token == "[SEP]"):
                separator_token = index
                break
        claim_starting_tokens=tokens[:5] #this is used only to print error message. dont worry too much about this.
        assert separator_token != 0
        for layer_index,per_layer_attention in enumerate(attention):
            if(layer_index==layer):
                for heads in per_layer_attention:
                    for head_index,per_head_attention in enumerate(heads):
                        if(head_index==head):
                            for_this_claim_what_is_the_total_attention_weight_that_came_from_other_tokens_in_claim=0
                            for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence = 0
                            for per_left_token_attention, token_column in zip(per_head_attention, tokens):
                                # Now go column wise. i.e for each word in column[0] of, what was the attention it got from other words.
                                # aggregate attention into two classes now if the index is before separator_token_index or not

                                #go to the next claim evidence pair once you hit the end of claim.
                                # This is because right now we care only about attention on claim words.
                                # Eventually it must be for all words, cross sentence and in-sentence
                                if (token_column == "[SEP]"):
                                    break

                                for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_claim = 0
                                for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence = 0

                                assert len(per_left_token_attention.data.tolist()) == len(tokens)
                                for index2,(weight, token) in enumerate(zip(per_left_token_attention.data.tolist(), tokens)):
                                    #ignore the attention placed on [SEP]. Bert places way too much attention on it.
                                    # + we have enough problems without it + we only care about claim and evidence tokens here, not separateor token
                                    if(token=="[SEP]"):
                                        continue

                                    if(index2>separator_token):
                                        for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence += weight
                                    else:
                                        for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_claim += weight

                                percent_attention_from_evidence_tokens=for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence*100/(for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence+for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_claim)
                                logger.debug(f"for the token : {token_column} in claim {percent_attention_from_evidence_tokens} percentage of attention weights that came from "
                                      f"came from evidence and rest from tokens in claim itself")
                                for_this_claim_what_is_the_total_attention_weight_that_came_from_other_tokens_in_claim+=for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_claim
                                for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence+=for_each_token_in_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence

                            overall_attention_from_evidence_tokens_percentage = for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence * 100 / (for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence + for_this_claim_what_is_the_total_attention_weight_that_came_from_other_tokens_in_claim)
                            logger.debug(
                                f"\nfor this claim which starts with {claim_starting_tokens} percentage of attention weights that came from "
                                f"came from evidence is {overall_attention_from_evidence_tokens_percentage} and the rest from tokens in claim itself")
        return for_this_claim_what_is_the_total_attention_weight_that_came_from_tokens_in_evidence,for_this_claim_what_is_the_total_attention_weight_that_came_from_other_tokens_in_claim



    def add_attentions_per_tokens(tokens_list, weights_list):
        weight_to_add = 0
        for (token) in zip(tokens_list, weights_list):
            for (token, weight) in zip(tokens_list, weights_list):
                weight_to_add = weight_to_add + weight
        assert weight_to_add !=0
        return weight_to_add

    def find_attention_percentage_across_claim_ev_both_directions(attention, tokens, layer, head):

        logger.debug(f"getting into fn find_attention_percentage_across_claim_ev_both_directions ")

        # first go through the entire tokens and find where first [SEP] is. that is where the
        # claim ends and evidence starts
        separator_token_index = 0
        for index, token in enumerate(tokens):
            if (token == "[SEP]"):
                separator_token_index = index
                break
        claim_starting_tokens = tokens[:5]  # this is used only to print error message. dont worry too much about this.
        assert separator_token_index != 0

        tokens_claim = tokens[:separator_token_index]

        #get all the tokens in evidence except that of [SEP]
        tokens_evidence = tokens[(separator_token_index+1):]

        for layer_index, per_layer_attention in enumerate(attention):
            if layer_index == layer:
                for heads in per_layer_attention:
                    for head_index, per_head_attention in enumerate(heads):
                        if head_index == head:
                            total_attention_weight_that_came_from_in_sent = 0
                            total_attention_weight_that_came_from_across_sent = 0


                            total_attention_weight_that_came_from_tokens_in_same_data_subpoint = 0  # a data sub point means claim or evidence
                            total_attention_weight_that_came_from_tokens_from_data_subpoint_across = 0  # note sentence can

                            # each_attention_column is the weight for a given token (lets call it x)/that came from all the other
                            # tokens in the entire datapoint (claim and evidence combined). So each_attention_column[0] is the weight that came to x from 1st token
                            # (called token_column here), each_attention_column[1] from second token etc etc

                            for index_attention_column, (each_attention_column,token_column) in enumerate(zip(per_head_attention,tokens)):

                                #ignore weights to or from bert specific tokens
                                # if (token_column == "[SEP]") or (token_column == "[CLS]") or (token_column == "[PAD]"):
                                #     continue

                                for index_token, (token_row) in enumerate(tokens):
                                    logger.debug(
                                        f"---------\n"
                                        f"the column token we are looking at is {token_column} and the row token is {token_row} ")


                                    # if (token_row == "[SEP]") or (token_row == "[CLS]") or (token_row == "[PAD]"):
                                    #     continue

                                    # if token is in claim and attention is coming from a token in evidence, increase cross_sentence weight, else increase in_sentence weight. and vice versa
                                    # token we are looking at (index_attention_column)is in claim, and the attention is coming from a token in claim
                                    if(index_token < separator_token_index) and (index_attention_column < separator_token_index):
                                        logger.debug(
                                            f"the column token we are looking at is {token_column} which is in claim and the weight is coming from the token {token_row} which also is in claim")
                                        total_attention_weight_that_came_from_tokens_in_same_data_subpoint+= each_attention_column[index_token].item()

                                    # token we are looking at (index_attention_column)is in evidence, and the attention is coming from a token in evidence
                                    if (index_token > separator_token_index) and (
                                            index_attention_column > separator_token_index):
                                        logger.debug(
                                            f"the column token we are looking at is {token_column} which is in evidence and the weight is coming from the token {token_row} which also is in evidence")

                                        total_attention_weight_that_came_from_tokens_in_same_data_subpoint += \
                                        each_attention_column[index_token].item()

                                    #token we are looking at (index_attention_column)is in claim, while the attention is coming from a token in evidence
                                    if (index_token > separator_token_index) and (
                                            index_attention_column < separator_token_index):
                                        logger.debug(
                                            f"the column token we are looking at is {token_column} which is in claim and the weight is coming from the token {token_row} which  is in evidence")

                                        total_attention_weight_that_came_from_tokens_from_data_subpoint_across += \
                                            each_attention_column[index_token].item()

                                    # token we are looking at (index_attention_column)is in evidence, while the attention is coming from a token in claim
                                    if (index_token < separator_token_index) and (
                                            index_attention_column > separator_token_index):
                                        logger.debug(
                                            f"the column token we are looking at is {token_column} which is in evidence and the weight is coming from the token {token_row} which  is in claim")

                                        total_attention_weight_that_came_from_tokens_from_data_subpoint_across += \
                                            each_attention_column[index_token].item()


                            logger.debug(
                                    f"total_attention_weight_that_came_from_tokens_in_same_data_subpoint= {total_attention_weight_that_came_from_tokens_in_same_data_subpoint} ")

                            logger.debug(
                                    f"total_attention_weight_that_came_from_tokens_from_data_subpoint_across= {total_attention_weight_that_came_from_tokens_from_data_subpoint_across} ")

                            percent_attention_from_evidence_tokens = total_attention_weight_that_came_from_tokens_from_data_subpoint_across * 100 / (
                                    total_attention_weight_that_came_from_tokens_from_data_subpoint_across + total_attention_weight_that_came_from_tokens_in_same_data_subpoint)

                            logger.debug(
                                f"\nfor this claim which starts with {claim_starting_tokens} percentage of attention weights that came from "
                                    f"came from cross sentence both directions is {percent_attention_from_evidence_tokens} and the rest from tokens in sentence itself")




        return total_attention_weight_that_came_from_tokens_from_data_subpoint_across, total_attention_weight_that_came_from_tokens_in_same_data_subpoint

    def find_aggregate_attention_per_token(attention, tokens, dict_token_attention, layer, head):
        for layer_index,per_layer_attention in enumerate(attention):
            if(layer_index==layer):
                for heads in per_layer_attention:
                    for head_index,per_head_attention in enumerate(heads):
                        if(head_index==head):
                            for per_left_token_attention in per_head_attention:
                                assert len(per_left_token_attention.data.tolist())==len(tokens)
                                for weight,token in zip(per_left_token_attention.data.tolist(),tokens):
                                    current_attention_weight=dict_token_attention.get(token, -1)
                                    #if the token already exists, increase its attention weight. else set its attention weight for the first time
                                    if not current_attention_weight==-1:
                                        current_attention_weight+=weight
                                        dict_token_attention[token]= current_attention_weight
                                    else:
                                        dict_token_attention[token] = weight






    if training_args.do_train_1student_1teacher:
        trainer = StudentTeacherTrainer(
            tokenizer_delex,
            tokenizer_lex,
            models={"teacher": model_teacher, "student": model_student},
            args=training_args,
            train_datasets={"combined": None},
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            eval_compute_metrics=dev_compute_metrics,
            test_compute_metrics=test_compute_metrics
        )
    else:
        trainer = OneModelAloneTrainer(
            tokenizer_delex,
            tokenizer_lex,
            model=model_for_bert,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            eval_compute_metrics=dev_compute_metrics,
            test_compute_metrics=test_compute_metrics
        )

    tokenizer_to_use=None
    #best student teacher trained (aka combined) models

    if training_args.do_train_1student_1teacher:
        url = 'https://osf.io/ht9gb/download'  # celestial-sun-1042 combined trained model- githubsha 21dabe wandb_celestial_sun1042 best_cd_acc_fnc_score_71.89_61.12



    if(training_args.task_type=="lex"):
      print("found use_lex==True")
      url = 'https://osf.io/fp89k/download' #trained_model_lex_helpful_vortex_1002_trained_model_afterepoch1_accuracy70point21percent..bin

    model_path=None
    #url = 'https://osf.io/uspm4/download'  # link to best delex trained model-this gave 55.69 in cross domain fnc score and 54.04 for cross domain accuracy
    # refer:https://tinyurl.com/y5dyshnh for further details regarding accuracies




    # in laptop we dont want to download model everytime. will load from a pre-downloaded-location
    if(training_args.machine_to_run_on=="laptop"):
        device = torch.device('cpu')
        if training_args.do_train_1student_1teacher:
            model_path = "/Users/mordor/research/huggingface/mithun_scripts/trained_models/student_teacher_trained_model.bin"
        if (training_args.task_type=="lex"):
            model_path = "/Users/mordor/research/huggingface/mithun_scripts/trained_models/lex_trained_model.bin"
    else:
        model_path = wget.download(url)
        device = torch.device("cuda:0")




    if training_args.do_train_1student_1teacher:
        print("found that if use_student_teacher==True:")
        model_for_bert=model_student
        tokenizer_to_use=tokenizer_delex

    if training_args.task_type=="lex":
        tokenizer_to_use=tokenizer_lex


    
    assert model_path is not None
    assert len(model_path)>0
    if(training_args.machine_to_run_on=="laptop"):
        model_for_bert.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model_for_bert.load_state_dict(torch.load(model_path))


    

    
    sentence_a = "Soon Marijuana May Lead to Ticket , Not Arrest , in New York"
    sentence_b = "After campaigning on a promise to reform stop-and-frisk , Mayor Bill de Blasio is set to launch his most significant effort to address the issues raised by the policy ."

    if training_args.do_train_1student_1teacher:
      sentence_a="Soon Marijuana May Lead to Ticket , Not Arrest , in governmentC1 "
      sentence_b = "After campaigning on a promise to reform stop-and-frisk , Mayor politicianE1 is set to launch his most significant effort to address the issues raised by the policy ."

    assert model_for_bert is not None
    assert url is not None
    assert len(url)>0
    assert tokenizer_to_use is not None
    assert len(sentence_a)>0
    assert len(sentence_b)>0

    
    inputs = tokenizer_to_use.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
    token_type_ids = inputs['token_type_ids']
    input_ids = inputs['input_ids']
    '''
    structure of attention=attention[layer][0][num_heads][seq_length][seq_length]
    so to get the attention weight given by layer 2, head 3, to token 7 (of all the x tokens when both sequences were 
    combined), when the home pointer is in token 5, you should do:
    
    attention[1][0][6][4] 
    
    '''




    dict_layer_head = get_cross_claim_evidence_percentage_attention_weights(test_dataset,model_for_bert,tokenizer_to_use)








def main(argv):

    base_file_path=""
    machine_to_run_on=""
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print("ERroror.")
        'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            base_file_path= arg
        if opt == '-o':
            machine_to_run_on= arg



    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    configs = read_and_merge_config_entries(base_file_path,machine_to_run_on)

    configs_split = configs.split()


    assert len(configs_split)>0


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=configs_split)

    training_args.output_dir=training_args.output_dir.replace("%20"," ")
    data_args.data_dir=data_args.data_dir.replace("%20"," ")
    training_args.toy_data_dir_path=training_args.toy_data_dir_path.replace("%20"," ")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    run_loading_and_testing(model_args, data_args, training_args)
   


if __name__ == "__main__":


    main(sys.argv[1:])





