import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModel
from utils.loss import FocalLoss


class IBModel(nn.Module):
    """
    Model implementation of paper: 'An Improved Baseline for Sentence-level Relation Extraction'
    with some customizations added to match the performance on KLUE RE task
    https://arxiv.org/abs/2102.01373
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
