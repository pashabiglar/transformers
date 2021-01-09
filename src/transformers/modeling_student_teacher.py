from torch import nn
from src.transformers.modeling_auto import AutoModel
from torch.nn import CrossEntropyLoss


class OneTeacherOneStudent(nn.Module):
    def __init__(self,config,model_name_or_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.model_teacher = Teacher(config)
        self.model_student = Student(config)
        self.t1 = Teacher(config)

    def forward(self, teacher_data, student_data):
        _, tencoded = self.bert(teacher_data)
        rt1 = self.t1(tencoded)
        return rt1

class Teacher(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,input_teacher):
        outputs=self.classifier(input_teacher)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), input_teacher["labels"].view(-1))
        logits= outputs[2]
        return loss, logits


class Student(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,input_teacher):
        outputs=self.classifier(input_teacher)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), input_teacher["labels"].view(-1))
        logits= outputs[2]
        return loss, logits