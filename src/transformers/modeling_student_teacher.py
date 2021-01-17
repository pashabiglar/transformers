import torch
from torch import nn
from transformers.modeling_auto import AutoModel
from torch.nn import CrossEntropyLoss
import sys,os
import logging
logger = logging.getLogger(__name__)
WEIGHTS_NAME_STUB = "pytorch_model"
import git

class OneTeacherOneStudent(nn.Module):
    def __init__(self,config,model_name_or_path):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.model_teacher = Teacher(config)
        self.model_student = Student(config)

    def save_pretrained(self, save_directory):
            """
            Save a model and its configuration file to a directory, so that it can be re-loaded using the
            `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

            Arguments:
                save_directory (:obj:`str`):
                    Directory to which to save. Will be created if it doesn't exist.
            """
            if os.path.isfile(save_directory):
                logger.error("Provided path ({}) should be a directory, not a file".format(save_directory))
                return
            os.makedirs(save_directory, exist_ok=True)

            # Only save the model itself if we are using distributed training
            model_to_save = self.module if hasattr(self, "module") else self

            # Attach architecture to the config
            #commenting this out since we dont care about saving aconfig in student teacher
            #model_to_save.config.architectures = [model_to_save.__class__.__name__]

            # If we save using the predefined names, we can load using `from_pretrained`

            git_details = self.get_git_info()

            model_file_name = WEIGHTS_NAME_STUB + "_" + git_details['repo_short_sha'] + ".bin"

            output_model_file = os.path.join(save_directory, model_file_name)

            # if getattr(self.config, "xla_device", False):
            #     import torch_xla.core.xla_model as xm
            #
            #     if xm.is_master_ordinal():
            #         # Save configuration file
            #         model_to_save.config.save_pretrained(save_directory)
            #     # xm.save takes care of saving only from master
            #     xm.save(model_to_save.state_dict(), output_model_file)
            # else:
            #     model_to_save.config.save_pretrained(save_directory)
            torch.save(model_to_save.state_dict(), output_model_file)

            logger.info("Model weights saved in {}".format(output_model_file))

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

    def forward(self, teacher_data, student_data):
        loss_student=logits_student=loss_teacher=logits_teacher=None

        #at the evaluation phase, teacher_data will be None
        if(teacher_data) is not None:
            input_ids=teacher_data.get('input_ids',None)[0]
            inputs_embeds = teacher_data.get('inputs_embeds', None),
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            attention_mask = teacher_data.get('attention_mask', None)
            if attention_mask is None:
                attention_mask = torch.ones(input_ids[0].shape(), device=device)
            _, tencoded = self.bert(teacher_data.get('input_ids',None),
                                    attention_mask=attention_mask,
                            token_type_ids = teacher_data.get('token_type_ids',None),
                            position_ids = teacher_data.get('position_ids',None),
                            head_mask = teacher_data.get('head_mask',None),
                            inputs_embeds = teacher_data.get('inputs_embeds',None),
                            output_attentions = teacher_data.get('output_attentions',None),
                            output_hidden_states = teacher_data.get('output_hidden_states',None),
                            return_tuple = teacher_data.get('return_tuple',None)
            )
            tencoded_dropout = self.dropout(tencoded)

            loss_teacher, logits_teacher = self.model_teacher(tencoded_dropout, teacher_data['labels'])
        if (student_data) is not None:
            input_ids = student_data.get('input_ids', None)[0]
            inputs_embeds = student_data.get('inputs_embeds', None),
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            attention_mask = student_data.get('attention_mask', None)
            if attention_mask is None:
                attention_mask = torch.ones(input_ids[0].shape(), device=device)
            _, sencoded = self.bert(student_data.get('input_ids', None),
                                attention_mask=attention_mask,
                                token_type_ids=student_data.get('token_type_ids', None),
                                position_ids=student_data.get('position_ids', None),
                                head_mask=student_data.get('head_mask', None),
                                inputs_embeds=student_data.get('inputs_embeds', None),
                                output_attentions=student_data.get('output_attentions', None),
                                output_hidden_states=student_data.get('output_hidden_states', None),
                                return_tuple=student_data.get('return_tuple', None)
                                )

            sencoded_dropout = self.dropout(sencoded)

            loss_student, logits_student = self.model_student(sencoded_dropout,student_data['labels'])

        out={"output_teacher":[loss_teacher,logits_teacher],
             "output_student":[loss_student,logits_student]}
        return out

class Teacher(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

    def forward(self,input_teacher,labels):
        logits=self.classifier(input_teacher)
        loss_fct = CrossEntropyLoss()
        loss_output = loss_fct(logits, labels)
        return loss_output, logits

class Student(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_labels = config.num_labels

    def forward(self,input_student,labels):
        logits=self.classifier(input_student)
        loss_fct = CrossEntropyLoss()
        loss_output = loss_fct(logits, labels)
        return loss_output, logits
