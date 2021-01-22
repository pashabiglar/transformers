import logging
import math
import os
import re
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import  Callable, Dict, List, Optional, Tuple
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange
from .data.data_collator import DataCollator, default_data_collator, collate_batch_parallel_datasets,collate_batch_for_3_datasets
from .file_utils import is_apex_available, is_torch_tpu_available
from .modeling_utils import PreTrainedModel
from .optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    is_wandb_available,
    set_seed,
)
from .training_args import TrainingArguments


import git
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


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


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
    test_dataset =Optional[Dataset]
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
        models: [],
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
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None,None)
    ):

        """
        Trainer is a simple but feature-complete traininfg and eval loop for PyTorch,
        optimized for Transformers.
        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.list_all_models=[]
        for each_model in models:
            self.list_all_models.append(each_model.to(args.device))
        assert len(self.list_all_models)>0
        self.eval_dataset = eval_dataset
        self.lex_tokenizer=tokenizer_lex
        self.delex_tokenizer = tokenizer_delex

        ###for fnc score evaluation
        self.test_dataset = test_dataset
        self.test_compute_metrics = test_compute_metrics
        self.eval_compute_metrics = eval_compute_metrics
        self.compute_metrics = None

        #even though we train using multiple models, finally there is only one THE model which is trained.- usually the one called as student model.
        # It is called self.model. Rather, this will be the model which will finally be used to test on dev and test sets
        # it is initiated as null for now, but after training is over, this will get assigned the trained model.
        self.model=None

        self.args = args
        self.default_data_collator = default_data_collator
        self.data_collator = collate_batch_for_3_datasets
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

    def write_predictions_to_disk(self, plain_text, gold_labels, predictions_logits, file_to_write_predictions, test_dataset):
        predictions_argmaxes = np.argmax(predictions_logits, axis=1)
        sf=torch.nn.Softmax(dim=1)
        predictions_softmax=sf(torch.FloatTensor(predictions_logits))
        if self.is_world_master():
            with open(file_to_write_predictions, "w") as writer:
                logger.info(f"***** (Going to write Test results to disk at {file_to_write_predictions} *****")
                writer.write("index\t gold\tprediction_logits\t prediction_label\tplain_text\n")
                for index,(gold, pred_sf, pred_argmax, plain) in enumerate(zip(gold_labels, predictions_softmax,predictions_argmaxes, plain_text)):
                    gold_string = test_dataset.get_labels()[gold]
                    pred_string = test_dataset.get_labels()[pred_argmax]
                    writer.write("%d\t%s\t%s\t%s\t%s\n" % (index, gold_string, str(pred_sf.tolist()),pred_string,plain))


    def predict_given_trained_model(self, model, test_dataset):
        predictions_argmaxes = self.predict(test_dataset,model).predictions_argmaxes


    def predict(self, test_dataset: Dataset,model_to_test_with) -> PredictionOutput:
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

        return self.prediction_loop(test_dataloader,model_to_test_with, description="Prediction")

    def evaluate(self, model_to_test_with,eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
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

        output,plain_text = self.prediction_loop(eval_dataloader,model_to_test_with, description="Evaluation")
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

    def get_train_dataloader_for_parallel_datasets(self) -> DataLoader:
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
                "params": [p for n, p in self.lex_teacher_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.lex_teacher_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.lex_teacher_model_ema.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.lex_teacher_model_ema.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.delex_student_model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.delex_student_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },

        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def get_optimizers_for_multiple_models(
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
        optimizer_grouped_parameters = []

        #Weight decay ensures that the parameters don't explore/remain in a certain reasonable range. it is usually addeed to the loss function
        # so that it can also be minimized with loss function.
        # apply 0 weight decay (i.e dont decay if the parameter is bias or layer normalization term, for everything else do apply
        #the specified weight decay.
        #  )
        for each_model in self.list_all_models:
            per_model_parameters=[]
            parameters_that_use_weight_decay={
                "params": [p for n, p in each_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            }
            parameters_that_wont_use_weight_decay={
                "params": [p for n, p in each_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
            per_model_parameters.append(parameters_that_use_weight_decay)
            per_model_parameters.append(parameters_that_wont_use_weight_decay)

            #sum up all the parameters. This is needed since in student teacher model, we are sharing the loss/weights across all architectures
            optimizer_grouped_parameters += per_model_parameters

        assert len(optimizer_grouped_parameters) > 0
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler


    def get_optimizer(
        self, model,num_training_steps: int) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
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
            for each_model in self.list_all_models:
                wandb.watch(
                    each_model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
                )



    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)


    def get_plaintext_given_dataloader(self):
        pass
        #return eval_dataloader.dataset.features

    def evaluate_on_test_partition(self, model_to_test_with,test_dataset: Optional[Dataset] = None,) -> Dict[str, float]:
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



        output,plain_text = self.prediction_loop(eval_dataloader,model_to_test_with ,description="Evaluation")
        gold_labels = output.label_ids
        predictions = output.predictions

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())
        return output.metrics, plain_text,gold_labels,predictions

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

    def _intermediate_eval(self, datasets, epoch, output_eval_file, description,model_to_test_with):

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
                eval_result, plain_text, gold_labels, predictions = self.evaluate(model_to_test_with,eval_dataset=dataset)
            else:
                if "test" in description:
                    eval_result, plain_text,gold_labels,predictions = self.evaluate_on_test_partition(model_to_test_with,test_dataset=dataset)
            assert eval_result is not None

            if self.is_world_master():
                with open(output_eval_file, "a") as writer:
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)
        return eval_result,plain_text,gold_labels,predictions


    def _save(self,output_dir: Optional[str] = None):
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
        self, model: nn.Module, inputs: Dict[str, torch.Tensor, ], prediction_loss_only: bool
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

    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def get_git_info(self):
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

    def train_1teacher_1student(self,model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader_for_parallel_datasets()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        model_teacher = self.lex_teacher_model
        model_teacher_ema = self.lex_teacher_model_ema
        model_student = self.delex_student_model

        weight_consistency_loss = 1
        weight_classification_loss = 0.0875

        optimizer = None
        scheduler = None

        #these flags are used for testing purposes. IDeally when running in student teacher mode this should be
        # flag_run_both=True. Other two flags are to test by loading each of these models independently from within
        #the same trainer class
        flag_run_teacher_alone = False
        flag_run_student_alone = False
        flag_run_both = True



        if (flag_run_both):
            optimizer, scheduler = self.get_optimizers_for_student_teacher(num_training_steps=self.args.lr_max_value)
            assert optimizer is not None
            assert scheduler is not None
        else:
            if (flag_run_teacher_alone):
                optimizer, scheduler = self.get_optimizer(model_teacher,num_training_steps=self.args.lr_max_value)
                assert optimizer is not None
                assert scheduler is not None
            else:
                if (flag_run_student_alone):
                    optimizer, scheduler = self.get_optimizer(model_student,num_training_steps=self.args.lr_max_value)
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

        #empty out the file which stores intermediate evaluations
        output_dir_absolute_path=os.path.join(os.getcwd(),self.args.output_dir)
        dev_partition_evaluation_output_file_path = output_dir_absolute_path + "intermediate_evaluation_on_dev_partition_results_"+git_details['repo_short_sha']+".txt"
        # empty out the
        with open(dev_partition_evaluation_output_file_path, "w") as writer:
            writer.write("")

        test_partition_evaluation_output_file_path =  output_dir_absolute_path+ "intermediate_evaluation_on_test_partition_results_"+git_details['repo_short_sha']+".txt"
        # empty out the
        with open(test_partition_evaluation_output_file_path, "w") as writer:
            writer.write("")


        # empty out the file which stores intermediate evaluations
        predictions_on_test_file_path = output_dir_absolute_path + "predictions_on_test_partition_"+git_details['repo_short_sha']+".txt"
        with open(predictions_on_test_file_path, "w") as writer:
            writer.write("")

        best_fnc_score=0
        best_acc=0


        #for each epoch

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

            #for each batch
            for step, (input_lex,input_delex) in enumerate(epoch_iterator):
                logger.debug("just got inside for step in enumerate epoch_iterator. i.e for each batch")


                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                assert input_lex['labels'].tolist()==input_delex['labels'].tolist()



                #model returns # (loss), logits, (hidden_states), (attentions)
                tr_loss_lex,outputs_lex = self.get_classification_loss(model_teacher, input_lex, optimizer)
                tr_loss_lex_ema, outputs_lex_ema = self.get_classification_loss(model_teacher_ema, input_lex, optimizer)
                tr_loss_delex,outputs_delex = self.get_classification_loss(model_student, input_delex, optimizer)


                if(flag_run_both):
                    combined_classification_loss=tr_loss_lex+tr_loss_delex

                else:
                    if(flag_run_teacher_alone):
                        combined_classification_loss = tr_loss_lex
                    else:
                        if(flag_run_student_alone):
                            combined_classification_loss = tr_loss_delex


                logger.debug("finished getting classification loss")


                # outputs contains in that order # (loss), logits, (hidden_states), (attentions)-src/transformers/modeling_bert.py
                logits_lex = outputs_lex[1]
                logits_lex_ema = outputs_lex_ema[1]
                logits_delex=outputs_delex[1]

                #in the world of student teacher with ema, the consistency loss will be between the student and ema of something. that something
                #can be an emaof lex , ema of delex, or both
                #consistency_loss = self.get_consistency_loss(logits_lex_ema,logits_delex,"mse")
                consistency_loss = self.get_consistency_loss(logits_lex, logits_delex, "mse")

                if (flag_run_both):
                    combined_loss = (weight_classification_loss*combined_classification_loss) + (weight_consistency_loss*consistency_loss)
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

                tr_loss_lex_float+=tr_loss_lex.item()
                tr_loss_delex_float+=tr_loss_delex.item()

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
                    self.update_ema_variables(model_teacher, model_teacher_ema, self.args.weight_decay,
                                              self.global_step)

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
                        #update: in student teacher setting since there are way too many model words going on, we will ezxplicitly pass the model to save

                        if (flag_run_both):
                            if hasattr(model_teacher, "module"):
                                assert model_teacher.module is self.lex_teacher_model.module
                                assert model_teacher_ema.module is self.lex_teacher_model_ema.module
                                assert model_student.module is self.delex_student_model.module

                            else:
                                assert model_teacher is self.lex_teacher_model
                                assert model_teacher_ema is self.lex_teacher_model_ema
                                assert model_student is self.delex_student_model


                            self.model=model_teacher
                            output_dir = os.path.join(self.args.output_dir,
                                                      f"model_teacher_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                            assert self.model is not None
                            self.save_model(output_dir)

                            self.model = model_student
                            output_dir = os.path.join(self.args.output_dir,
                                                      f"model_student_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                            assert self.model is not None
                            self.save_model( output_dir)


                        else:
                            if (flag_run_teacher_alone):
                                if hasattr(model_teacher, "module"):
                                    assert model_teacher.module is self.lex_teacher_model.module
                                    assert model_teacher_ema.module is self.lex_teacher_model_ema.module
                                else:
                                    assert model_teacher is self.lex_teacher_model
                                    assert model_teacher_ema.module is self.lex_teacher_model_ema.module
                                self.model = model_teacher
                                output_dir = os.path.join(self.args.output_dir,
                                                          f"model_teacher_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                                self.save_model( output_dir)
                            else:
                                if (flag_run_student_alone):
                                    if hasattr(model_teacher, "module"):
                                        assert model_student.module is self.delex_student_model.module
                                    else:
                                        assert model_student is self.delex_student_model
                                self.model = model_student
                                output_dir = os.path.join(self.args.output_dir,
                                                          f"model_student_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                                self.save_model( output_dir)

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

            trained_model=None
            if (flag_run_both):
                trained_model =model_student
            else:
                if (flag_run_teacher_alone):
                    trained_model = model_teacher
                else:
                    if (flag_run_student_alone):
                        trained_model = model_student

            assert trained_model is not None

           
            dev_partition_evaluation_result,plain_text,gold_labels,predictions = self._intermediate_eval(datasets=self.eval_dataset,
                                                                      epoch=epoch,
                                                                      output_eval_file=dev_partition_evaluation_output_file_path,
                                                                      description="dev_partition",model_to_test_with=trained_model)

            test_partition_evaluation_result,plain_text,gold_labels,predictions_logits=self._intermediate_eval(datasets=self.test_dataset,
                                    epoch=epoch, output_eval_file=test_partition_evaluation_output_file_path, description="test_partition",model_to_test_with=trained_model)

            fnc_score_test_partition = test_partition_evaluation_result['eval_acc']['cross_domain_fnc_score']
            accuracy_test_partition = test_partition_evaluation_result['eval_acc']['cross_domain_acc']



            if fnc_score_test_partition>best_fnc_score:
                best_fnc_score=fnc_score_test_partition

                logger.info(f"found that the current fncscore:{fnc_score_test_partition} in epoch "
                            f"{epoch} beats the bestfncscore so far i.e ={best_fnc_score}. going to prediction"
                            f"on test partition and save that and model to disk")
                #if the accuracy or fnc_score_test_partition beats the highest so far, write predictions to disk

                self.write_predictions_to_disk(plain_text, gold_labels, predictions_logits, predictions_on_test_file_path,
                                               self.test_dataset)

                # Save model checkpoint
                self.model=trained_model
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
        return dev_partition_evaluation_result,test_partition_evaluation_result

    def train_multiple_teachers_1student(self, model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """

        train_dataloader = self.get_train_dataloader_for_parallel_datasets()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                    self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        weight_consistency_loss = 1
        weight_classification_loss = 0.0875

        optimizer = None
        scheduler = None
        optimizer, scheduler = self.get_optimizers_for_multiple_models(num_training_steps=self.args.lr_max_value)
        assert optimizer is not None
        assert scheduler is not None


        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()


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

        for each_model in self.list_all_models:
            each_model.zero_grad()

        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        git_details = self.get_git_info()

        # empty out the file which stores intermediate evaluations on dev
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

        # empty out the file which stores intermediate evaluations on test partition (which is not really test partition, but the dev of cross-domain dataset)
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
            #todo: dont hardcode the number 3. this should work for n models
            for step, (input_model1, input_model2, input_model3) in enumerate(epoch_iterator):
                logger.debug("just got inside for step in enumerate epoch_iterator. i.e for each batch")

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                assert input_model1['labels'].tolist() == input_model2['labels'].tolist() == input_model3['labels'].tolist()

                all_inputs=[input_model1,input_model2,input_model3]

                assert len(all_inputs) == len(self.list_all_models)



                combined_classification_loss = torch.zeros(1).to(device=self.args.device)
                all_models_outputs=[]
                for index,each_model in enumerate(self.list_all_models):
                    # model returns # (loss), logits, (hidden_states), (attentions)
                    tr_classification_loss, outputs_model = self.get_classification_loss(each_model, all_inputs[index], optimizer)
                    combined_classification_loss += tr_classification_loss
                    all_models_outputs.append(outputs_model)
                assert index == (len(self.list_all_models)-1)

                # calculate consistency loss:
                # consistency loss is the loss between logits of student model(model 1) and logits of other models
                # we are considering the second model (the one which reads data delexicalized in figerspecific format as the student model. ideally this could have been any). hence all_models_outputs[1]
                # the second [1] is because logits are the second entry in any output returned by this model
                logits_student = all_models_outputs[1][1]
                combined_consistency_loss = torch.zeros(1).to(device=self.args.device)

                assert len(all_models_outputs) > 0
                for index,each_model_output in enumerate(all_models_outputs):
                    if not index ==1:
                        # outputs contains in that order # (loss), logits, (hidden_states), (attentions)-src/transformers/modeling_bert.py
                        #so logits will be output[1]
                        logits_teacher = each_model_output[1]
                        consistency_loss = self.get_consistency_loss(logits_teacher, logits_student, "mse")
                        combined_consistency_loss += consistency_loss

                assert combined_classification_loss.item() > 0
                assert combined_consistency_loss.item() > 0
                #overall loss is a weighteed sum of classification and consistency losses
                combined_loss = (weight_classification_loss * combined_classification_loss) + (
                                weight_consistency_loss * combined_consistency_loss)
                combined_loss.backward()
                optimizer.step()
                scheduler.step()

                for each_model in self.list_all_models:
                    torch.nn.utils.clip_grad_norm_(each_model.parameters(), self.args.max_grad_norm)

                for each_model in self.list_all_models:
                        each_model.zero_grad()

                self.global_step += 1
                self.epoch = epoch + (step + 1) / len(epoch_iterator)





            trained_model = self.list_all_models[1]
            self.model = self.list_all_models[1]


            assert trained_model is not None
            assert self.model is not None

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
        # model outputs are always a list of things in a class in #
        # Bert(see that modeling_bert/BertForSequenceClassification returns a class SequenceClassifierOutput)
        loss = outputs[0]
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        return loss,outputs

    def get_logits(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) :
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
            self, logit1,logit2,loss_function):
        '''
        similar to _training_step however returns loss instead of doing .backward
        Args:

            model:
            inputs:
            optimizer:
        Returns:loss
        '''\


        if(loss_function=="mse"):
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
        self, dataloader: DataLoader, model_to_test_with,description: str, prediction_loss_only: Optional[bool] = None
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
        logger.debug("***** Running %s at epoch number:%s *****", description,self.epoch)
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
        plain_text_full=[]
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
        assert len(plain_text_full)== len(dataloader.dataset.features)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics),plain_text_full


class GlobalTrainer:
    """
    GlobalTrainer is a trainer which should be able to train anything including any number of models.
    i.e this class was created to get rid of having to have separate trainer classes for different architectures like student teacher architecture.
    Instead all the changes will be encompassed in the corresponding modeling_bert class.

    Left of the comments below are the same as that of the original Trainer class from hugging face: src/transformers/trainer/Trainer


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

    def __init__(
            self,
            tokenizer_delex,
            model: PreTrainedModel,
            args: TrainingArguments,
            train_datasets: {},
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            test_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            eval_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
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
        self.model = model
        self.eval_dataset = eval_dataset
        ###for fnc score evaluation
        self.test_dataset = test_dataset
        self.test_compute_metrics = test_compute_metrics
        self.eval_compute_metrics = eval_compute_metrics
        self.compute_metrics = None
        self.args = args
        #in the student teacher architechture we pass both lexicalized and delexicalized version of the dataset as a "parallel" dataset
        #where every data point has two parts.      Also typically you need the parallel dataset only for training.
        self.serial_data_collator =      default_data_collator
        self.parallel_dataset_data_collator =  collate_batch_parallel_datasets
        self.train_dataset_combined = train_datasets.get("combined")
        self.eval_dataset = eval_dataset
        self.compute_metrics = None
        self.prediction_loss_only = prediction_loss_only
        self.optimizer, self.lr_scheduler = optimizers
        self.delex_tokenizer=tokenizer_delex
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

        if not callable(self.parallel_dataset_data_collator) and callable(getattr(self.parallel_dataset_data_collator, "collate_batch", None)):
            self.parallel_dataset_data_collator = self.parallel_dataset_data_collator.collate_batch
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
            collate_fn=self.parallel_dataset_data_collator.collate_batch,
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
            collate_fn=self.parallel_dataset_data_collator,
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
            collate_fn=self.serial_data_collator
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
            collate_fn=self.serial_data_collator,
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

        # add both params instead of grouping so that weights are updated together for both models
        # params_teacher = [p for n, p in self.model.model_teacher.named_parameters() if
        #                   not any(nd in n for nd in no_decay)]
        # params_student = [p for n, p in self.model.model_student.named_parameters() if
        #                   not any(nd in n for nd in no_decay)]
        # optimizer_grouped_parameters = params_teacher + params_student

        #group both parameters instead of adding them. as of jan 2021 we are not sure which is a better approach.
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.model_teacher.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.model_student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
         ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon,weight_decay= self.args.weight_decay)
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
                self.model, log=os.getenv("WANDB_WATCH", "gradients"),
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
            output_all_models = model(None,inputs)
            #the outputs_teacher will be none since we are not passing any input for it.
            assert output_all_models is not None
            if has_labels:
                logits= output_all_models['output_student'][1]
                loss = output_all_models['output_student'][0]
                loss = loss.mean().item()
            else:
                loss = None
                logits = output_all_models['output_student'][1]

        #not sure wqhat this does. whatever it does it will fail for student teacher arch- mithun jan2021
        if self.args.past_index >= 0:
            self._past = output_all_models[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, logits.detach(), labels)

    def update_ema_variables(self, model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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

    def train(self, model_path: Optional[str] = None):
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

        model = self.model

        #these parameters were tuned and arrived at the best value for student teacher combination i.e 0.0875
        weight_consistency_loss = 1
        weight_classification_loss = 0.0875

        optimizer = None
        scheduler = None

        #lr_max_value is now used as a static value inside the scheduler. Earlier it was a dynamic value based on number of training steps, which was messing up accuracy after eveyr epoch
        optimizer, scheduler = self.get_optimizers_for_student_teacher(num_training_steps=self.args.lr_max_value)
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
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)


        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()

            model = torch.nn.DataParallel(model)


        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
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
        model.zero_grad()


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
                batch_iterator = tqdm(parallel_loader, desc="batches", disable=not self.is_local_master())
            else:
                logger.debug("found that is_torch_tpu_available is false")
                batch_iterator = tqdm(train_dataloader, desc="batches", disable=not self.is_local_master())

            # for each batch
            for step, (input_lex, input_delex) in enumerate(batch_iterator):
                logger.debug("just got inside for step in enumerate epoch_iterator. i.e for each batch")

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                assert input_lex['labels'].tolist() == input_delex['labels'].tolist()

                output_all_models = self.get_classification_output_1student_1teacher(model, input_lex, input_delex, optimizer)
                combined_classification_losses=torch.zeros(1).to(device=self.args.device)
                for model_name,output in output_all_models.items():
                    combined_classification_losses += output[0]


                #in this particular case we have teacher as the first model and student as the second. will change based on teh architectujre
                logits_lex_teacher = output_all_models['output_teacher'][1]
                logits_delex_student = output_all_models['output_student'][1]


                #find how far is student from the teacher and minimixe that also
                consistency_loss = self.get_consistency_loss(logits_lex_teacher, logits_delex_student, "mse")

                #combine both losses for backpropagation. each of these weight values has to be tuned
                combined_loss = (weight_classification_loss * combined_classification_losses) + (
                                weight_consistency_loss * consistency_loss)


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

                #get the loss value as float for logging purposes
                tr_loss_lex_float += output_all_models['output_teacher'][0].item()
                tr_loss_delex_float += output_all_models['output_student'][0].item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        len(batch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(batch_iterator)
                ):
                    logger.debug("got inside if condition for if(step+1. i.e last step in epoch)")

                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    logger.debug("just done with grad clipping)")

                if is_torch_tpu_available():
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                    logger.debug("just done withn optimixer.step)")
                scheduler.step()
                # self.update_ema_variables(model, model_teacher_ema, self.args.weight_decay,
                    #                           self.global_step)

                model.zero_grad()




                self.global_step += 1
                self.epoch = epoch + (step + 1) / len(batch_iterator)

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


                    if hasattr(model, "module"):
                        assert model.module is self.lex_teacher_model.module
                        assert model_teacher_ema.module is self.lex_teacher_model_ema.module
                        assert model_student.module is self.delex_student_model.module

                    else:
                        assert model is self.lex_teacher_model
                        assert model_teacher_ema is self.lex_teacher_model_ema
                        assert model_student is self.delex_student_model

                    self.model = model
                    output_dir = os.path.join(self.args.output_dir,
                                              f"model_teacher_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")


                    #not saving during training since we are saving the best model every time it gets better on dev
                    # assert self.model is not None
                    #self.save_model(output_dir)





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
                    batch_iterator.close()
                    break



            dev_partition_evaluation_result, plain_text, gold_labels, predictions = self._intermediate_eval(
                datasets=self.eval_dataset,
                epoch=epoch,
                output_eval_file=dev_partition_evaluation_output_file_path,
                description="dev_partition", model_to_test_with=model)

            test_partition_evaluation_result, plain_text, gold_labels, predictions_logits = self._intermediate_eval(
                datasets=self.test_dataset,
                epoch=epoch, output_eval_file=test_partition_evaluation_output_file_path, description="test_partition",
                model_to_test_with=model)

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
                self.model = model
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

        #commenting this out since in student teacher architecture our class is not inherited from pretrained model- even though we still load a pretrained model. we just bypassed that class since it was too complicated to inherit
        # if not isinstance(self.model, PreTrainedModel):
        #     raise ValueError("Trainer.model appears to not be a PreTrainedModel")
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

    def get_classification_losses(self,outputs) -> float:
        '''
        returns all losses of the model (even if the model is internally combined)
        Args:
            model:
            inputs:
            optimizer:
        Returns:loss
        '''

        #if the model happens to be an ensemble of multiple models, go through all the outputs and pull out losses of each model
        all_losses=[]
        for each_model_output in outputs:
            # model returns # (loss), logits, (hidden_states), (attentions)
            loss = each_model_output[0]
            all_losses.append(loss)

        assert len(all_losses) > 0


        return all_losses

    def get_logits(self, outputs):
        '''
        returns all logits/final outputs of each model (if its a combined model
        model outputs are usually in the order (loss), logits, (hidden_states), (attentions)
        Args:
            outputs:output when the combined model was run
        Returns:
            list of logits of each model
        '''

        # if the model happens to be an ensemble of multiple models, go through all the outputs and pull out logits of each model
        all_logits = []
        for each_model_output in outputs:
            # model returns # (loss), logits, (hidden_states), (attentions)
            all_logits.append(each_model_output.logits)

        assert len(all_logits) > 0


        return all_logits


    def get_classification_output(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        '''
        returns output of the model (even if the model is internally combined)
        Args:
            model:
            inputs:
            optimizer:
        Returns:loss
        '''
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs_teacher, outputs_student = model(**inputs)
        return outputs_teacher, outputs_student



    def get_classification_output_1student_1teacher(
            self, model: nn.Module, input_teacher: Dict[str, torch.Tensor], input_student: Dict[str, torch.Tensor],optimizer: torch.optim.Optimizer
    ) -> float:
        '''
        returns output of the model (even if the model is internally combined)
        Args:
            model:
            inputs:
            optimizer:
        Returns:loss
        '''
        model.train()
        for k, v in input_teacher.items():
            input_teacher[k] = v.to(self.args.device)
        for k, v in input_student.items():
            input_student[k] = v.to(self.args.device)
        output = model(input_teacher, input_student)
        return output



    def combine_all_losses(self, losses):
        '''
        combines each of the losses and returns it

        Args:
        list of losses
        Returns:loss
        '''

        # if the model happens to be an ensemble of multiple models, go through all the outputs and pull out losses of each model
        combined_losses = torch.empty(losses[0].shape)

        for each_model_loss in losses:
            combined_losses  += each_model_loss
        assert combined_losses is not None
        return combined_losses


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
        print(dataloader)

        

        # for each batch
        #for step, (input_lex, input_delex) in enumerate(batch_iterator):

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


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """
    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    test_dataset = Optional[Dataset]
    test_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    eval_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
            self,
            tokenizer_delex,
            tokenizer_lex,
            model: PreTrainedModel,
            args: TrainingArguments,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            test_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            eval_compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            prediction_loss_only=False,
            tb_writer: Optional["SummaryWriter"] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.
        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        self.data_collator = data_collator if data_collator is not None else default_data_collator
        self.train_dataset = train_dataset
        self.lex_tokenizer = tokenizer_lex
        self.delex_tokenizer = tokenizer_delex
        # pointing eval dataset to test dataset is a temporary hack on july 28th to fix reproducability issues.
        self.eval_dataset = eval_dataset
        ###for fnc score evaluation
        self.test_dataset = test_dataset
        self.test_compute_metrics = test_compute_metrics
        self.eval_compute_metrics = eval_compute_metrics
        self.compute_metrics = None
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
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

        output,plain_text = self.prediction_loop(eval_dataloader, model_to_test_with, description="Evaluation")
        gold_labels = output.label_ids
        predictions = output.predictions

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return  output.metrics, plain_text,gold_labels,predictions

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
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator
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
            collate_fn=self.data_collator
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
            collate_fn=self.data_collator,
        )

        return data_loader

    def get_optimizers_student_teacher(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        update: will combine parameters
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.lex_teacher_model.named_parameters() if not any(nd in n for nd in no_decay)],
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
                "params": [p for n, p in self.delex_student_model.named_parameters() if any(nd in n for nd in no_decay)],
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
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )


    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

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

        output,plain_text = self.prediction_loop(eval_dataloader, model_to_test_with, description="Evaluation")
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
            plain_text=None
            gold_labels=None
            predictions=None
            if "dev" in description:
                eval_result,plain_text, gold_labels, predictions  = self.evaluate(model_to_test_with, eval_dataset=dataset)
            else:
                if "test" in description:
                    eval_result, plain_text, gold_labels, predictions = self.evaluate_on_test_partition(model_to_test_with, test_dataset=dataset)
            assert eval_result is not None

            if self.is_world_master():
                with open(output_eval_file, "a") as writer:
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
            eval_results.update(eval_result)
        return eval_result, plain_text,gold_labels,predictions

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
        self, model: nn.Module, inputs: Dict[str, torch.Tensor, ], prediction_loss_only: bool
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

    def train(self,model_path: Optional[str] = None):
        """
        Main training entry point.
        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs


        optimizer, scheduler = self.get_optimizers(lr_max_value=self.args.lr_max_value)

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

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
            logger.info(f"fouund that n_gpu>1")
            logger.info(
                f"found that self.args.Nn_gpu >1. going to exit.")
            import sys
            sys.exit()

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            print("fouund that self.args.local_rank != -1:. going to quit")
            logger.info(f"self.args.local_rank != -1. going to quit")
            # import sys
            # sys.exit(1)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
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
        model.zero_grad()

        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master()
        )

        #empty out the file which stores intermediate evaluations
        output_dir_absolute_path = os.path.join(os.getcwd(), self.args.output_dir)
        dev_partition_evaluation_output_file_path = output_dir_absolute_path + "intermediate_evaluation_on_dev_partition_results.txt"
        # empty out the
        with open(dev_partition_evaluation_output_file_path, "w") as writer:
            writer.write("")

        test_partition_evaluation_output_file_path = output_dir_absolute_path + "intermediate_evaluation_on_test_partition_results.txt"
        # empty out the
        with open(test_partition_evaluation_output_file_path, "w") as writer:
            writer.write("")


        # empty out the file which stores intermediate evaluations
        predictions_on_test_file_path = output_dir_absolute_path + "predictions_on_test_partition.txt"
        with open(predictions_on_test_file_path, "w") as writer:
            writer.write("")

        best_fnc_score=0
        best_acc=0

        #for each epoch
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="batches", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="batches", disable=not self.is_local_master())

            #for each batch
            for step, (input_lex) in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue


                #model returns # (loss), logits, (hidden_states), (attentions) in that order
                tr_loss_lex,outputs_lex = self.get_classification_loss(model, input_lex, optimizer)



                if self.args.fp16:
                    with amp.scale_loss(tr_loss_lex, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    tr_loss_lex.backward()

                tr_loss_lex_float+=tr_loss_lex.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)


                    if is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()

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
                        if hasattr(model, "module"):
                            assert model.module is self.lex_teacher_model
                        else:
                            assert model is self.lex_teacher_model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"model_teacher_{PREFIX_CHECKPOINT_DIR}-{self.global_step}")
                        self.save_model(output_dir)


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

            dev_partition_evaluation_result,plain_text,gold_labels,predictions = self._intermediate_eval(datasets=self.eval_dataset,
                                                                      epoch=epoch,
                                                                      output_eval_file=dev_partition_evaluation_output_file_path,
                                                                      description="dev_partition",
                                                                      model_to_test_with=model)

            test_partition_evaluation_result, plain_text,gold_labels,predictions = self._intermediate_eval(datasets=self.test_dataset,
                                                                       epoch=epoch,
                                                                       output_eval_file=test_partition_evaluation_output_file_path,
                                                                       description="test_partition",
                                                                       model_to_test_with=model)

            fnc_score_test_partition = test_partition_evaluation_result['eval_acc']['cross_domain_fnc_score']
            accuracy_test_partition = test_partition_evaluation_result['eval_acc']['cross_domain_acc']
            self.write_predictions_to_disk(plain_text, gold_labels, predictions, predictions_on_test_file_path,self.test_dataset)


            if fnc_score_test_partition > best_fnc_score:
                best_fnc_score = fnc_score_test_partition

                logger.info(f"found that the current fncscore:{fnc_score_test_partition} in epoch "
                            f"{epoch} beats the bestfncscore so far i.e ={best_fnc_score}. going to prediction"
                            f"on test partition and save that and model to disk")
                # if the accuracy or fnc_score_test_partition beats the highest so far, write predictions to disk
                self.write_predictions_to_disk(plain_text, gold_labels, predictions, predictions_on_test_file_path,
                                               self.test_dataset)

                # Save model checkpoint
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
                f"********************************end of epoch {epoch}************************************************************************")
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        # Note: returning the dev and test partition evaluation results to be used in test cases
        # Also note that all performance evaluation measures by now are written to disk in the log file.
        # Todo: the assumption here is that the testing/testcase will be run for 1 epoch only.  Ideally have to return the best dev and test partition scores and should work for any epoch
        return dev_partition_evaluation_result,test_partition_evaluation_result


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

        return loss,outputs

    def get_logits(
            self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) :
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
            self, logit1,logit2,loss_function):
        '''
        similar to _training_step however returns loss instead of doing .backward
        Args:
            model:
            inputs:
            optimizer:
        Returns:loss
        '''

        if(loss_function=="mse"):
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

    def get_optimizers(
            self, lr_max_value: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
        # def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=lr_max_value)
        return optimizer, scheduler

    def prediction_loop(
            self, dataloader: DataLoader, model_to_test_with,description: str, prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
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
            logger.debug("***** Running %s at epoch number:%s *****", description,self.epoch)
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



          
      