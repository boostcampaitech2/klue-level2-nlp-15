import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from utils.loss import FocalLoss


class IBModel(nn.Module):
    """
    Model implementation of paper: 'An Improved Baseline for Sentence-level Relation Extraction'
    with some customizations added to match the performance on KLUE RE task
    https://arxiv.org/abs/2102.01373 | https://github.com/wzhouad/RE_improved_baseline/blob/main/model.py
    """

    def __init__(self, model_name, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = FocalLoss(gamma=1.0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, config.num_labels),
        )

    @autocast()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        ss=None,
        se=None,
        es=None,
        ee=None,
    ):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        es_emb = pooled_output[idx, es]
        h = torch.cat((ss_emb, es_emb), dim=-1)
        logits = self.classifier(h)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs


class StartTokenWithCLSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = "klue/roberta-large"
        self.bert_model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.hidden_size = 1024
        self.num_labels = 30
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        special_tokens_dict = {
            "additional_special_tokens": [
                "[SUB:ORG]",
                "[SUB:PER]",
                "[/SUB]",
                "[OBJ:DAT]",
                "[OBJ:LOC]",
                "[OBJ:NOH]",
                "[OBJ:ORG]",
                "[OBJ:PER]",
                "[OBJ:POH]",
                "[/OBJ]",
            ]
        }

        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print("num_added_tokens:", num_added_tokens)

        self.bert_model.resize_token_embeddings(len(self.tokenizer))

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(3 * self.hidden_size, self.hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.hidden_size, self.num_labels),
        )

    def forward(self, item):
        input_ids = item["input_ids"]
        token_type_ids = item["token_type_ids"]
        attention_mask = item["attention_mask"]
        sub_token_index = item["sub_token_index"]
        obj_token_index = item["obj_token_index"]
        out = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        h = out.last_hidden_state
        batch_size = h.shape[0]

        stack = []

        for i in range(batch_size):
            stack.append(
                torch.cat([h[i][0], h[i][sub_token_index[i]], h[i][obj_token_index[i]]])
            )

        stack = torch.stack(stack)

        out = self.classifier(stack)
        return out


class FCLayer(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""

    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""

    def __init__(
        self,
        model_name: str = "klue/roberta-large",
        num_labels: int = 30,
        dropout_rate: float = 0.1,
        special_tokens_dict: dict = None,
        is_train: bool = True,
    ):
        super(RBERT, self).__init__()

        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        self.backbone_model = AutoModel.from_pretrained(model_name, config=config)
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels

        # add special tokens
        self.special_tokens_dict = special_tokens_dict
        self.backbone_model.resize_token_embeddings(len(tokenizer))

        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids,
        attention_mask,
        subject_mask=None,
        object_mask=None,
        labels=None,
    ):

        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs[
            "pooler_output"
        ]  # [CLS] token's hidden featrues(hidden state)

        # hidden state's average in between entities
        # print(sequence_output.shape, subject_mask.shape)
        e1_h = self.entity_average(
            sequence_output, subject_mask
        )  # token in between subject entities ->
        e2_h = self.entity_average(
            sequence_output, object_mask
        )  # token in between object entities

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(
            pooled_output
        )  # [CLS] token -> hidden state | green on diagram
        e1_h = self.entity_fc_layer(
            e1_h
        )  # subject entity's fully connected layer | yellow on diagram
        e2_h = self.entity_fc_layer(
            e2_h
        )  # object entity's fully connected layer | red on diagram

        # Concat -> fc_layer / [CLS], subject_average, object_average
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        return logits

        # WILL USE FOCAL LOSS INSTEAD OF MSELoss and CrossEntropyLoss
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #     outputs = (loss,) + outputs

        #  return outputs  # (loss), logits, (hidden_states), (attentions)
