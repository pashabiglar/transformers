from torch import nn
from transformers.modeling_auto import AutoModel
from torch.nn import CrossEntropyLoss
import sys

class OneTeacherOneStudent(nn.Module):
    def __init__(self,config,model_name_or_path):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.model_teacher = Teacher(config)
        self.model_student = Student(config)


    def forward(self, teacher_data, student_data):
        _, tencoded = self.bert(teacher_data.get('input_ids',None),
                        attention_mask = teacher_data.get('attention_mask',None),
                        token_type_ids = teacher_data.get('token_type_ids',None),
                        position_ids = teacher_data.get('position_ids',None),
                        head_mask = teacher_data.get('head_mask',None),
                        inputs_embeds = teacher_data.get('inputs_embeds',None),
                        output_attentions = teacher_data.get('output_attentions',None),
                        output_hidden_states = teacher_data.get('output_hidden_states',None),
                        return_tuple = teacher_data.get('return_tuple',None)
        )
        _, sencoded = self.bert(student_data.get('input_ids', None),
                                attention_mask=student_data.get('attention_mask', None),
                                token_type_ids=student_data.get('token_type_ids', None),
                                position_ids=student_data.get('position_ids', None),
                                head_mask=student_data.get('head_mask', None),
                                inputs_embeds=student_data.get('inputs_embeds', None),
                                output_attentions=student_data.get('output_attentions', None),
                                output_hidden_states=student_data.get('output_hidden_states', None),
                                return_tuple=student_data.get('return_tuple', None)
                                )
        tencoded_dropout=self.dropout(tencoded)
        sencoded_dropout = self.dropout(sencoded)

        loss_teacher, logits_teacher = self.model_teacher(tencoded_dropout, teacher_data['labels'])
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
