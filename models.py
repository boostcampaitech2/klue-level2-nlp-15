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
