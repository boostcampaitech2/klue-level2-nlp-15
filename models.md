### 안영진

```python
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

""" Define Custom Model -> will later allocated to models.py """
class CustomModel(nn.Module):
    def __init__(
        self,
        model_name:str,
        num_labels:int=30,
        num_multi_labels:int=29,
        input_size:int=768,
        output_size:int=768,
        num_rnn_layers:int=3,
        dropout_rate:float=0,
        is_train:bool=True
        ):

        super(CustomModel, self).__init__()
        self.model_name = model_name
        self.backbone_model = AutoModel.from_pretrained(self.model_name)

        # add bidrectional gru (multiple) layers in the end
        self.gru = nn.GRU(
            # set as BERT model's hidden size, not as an integer: flexible for different models
            input_size=self.backbone_model.config.hidden_size,
            hidden_size=output_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_rnn_layers,
            dropout=dropout_rate
            )

        # define fully connected layers with different purposes
        # gru layers' output size is 2*output_size, since it is bidirectional
        self.fc = nn.Linear(in_features=output_size*2, out_features=num_labels)
        self.fc_muiltihead = nn.Linear(in_features=output_size*2, out_features=3) # [no_relation, org, per]
        self.fc_multitask_1 = nn.Linear(in_features= output_size*2, out_features=2) # [no_relation, yes_relation]
        self.fc_multitask_2 = nn.Linear(in_features= output_size*2, out_features=num_multi_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        backbone_output = self.backbone_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # add gru layer
        gru_output = self.gru(backbone_output[0])
        # flatten the output
        gru_output = gru_output.view(gru_output.size(0), -1)
        # input as fully connected layers
        fc_output = self.fc(gru_output)
        return fc_output
```

```python
""" R-BERT: https://github.com/monologg/R-BERT """
class FCLayer(nn.Module):
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
    def __init__(
        self,
        model_name:str,
        num_labels:int=30,
        dropout_rate:float=0,
        is_train:bool=True
        ):
        super(RBERT, self).__init__()

        config = AutoConfig.from_pretrained(model_name)


        self.backbone_model = AutoModel.from_pretrained(model_name, config=config)
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        config.num_labels = num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, self.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, self.dropout_rate)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

    def entity_average(hidden_output, e_mask):
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

    def forward(self, model_name, input_ids, attention_mask, token_type_ids=None, labels=None, e1_mask=None, e2_mask=None):

        if "roberta" in model_name.lower():
            outputs = self.backbone_model(input_ids = input_ids, attention_mask = attention_mask)
        else:
            outputs = self.bert(
                input_ids = input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = logits

        return outputs

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

```

### 김현수

```python
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig, BertModel, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

""" 실...패....ㅠㅠ  업데이트 중 """
class CustomBert(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.init_weights()
        self.tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls = outputs[1]
        hidden_states = outputs[2]
        mask_index = []
        for i in range(len(input_ids)):
            mask_index.append(torch.where(input_ids[i] == self.tokenizer.mask_token_id)[0].tolist())
        #mask_index = torch.where(input_ids[0] == self.tokenizer.mask_token_id)
        token_embeddings = torch.stack(hidden_states, dim=0)

        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2,3)
        token_vecs_sum = []
        for token in token_embeddings:
           sum_vec = torch.sum(token, dim=0)
           token_vecs_sum.append(sum_vec)
        out_vecs = []

        for i in range(len(token_vecs_sum)):
            tmp = token_vecs_sum[i]
            out_vec = torch.cat([tmp[j].unsqueeze(0) for j in mask_index[i]], dim = -1)
            out_vecs.append(out_vec.squeeze(0).tolist())
        out_vecs = torch.tensor(out_vecs, device= torch.device("cuda"))
        '''
        for i in range(len(mask_index)):
            out_vec = torch.cat(
                [token_vecs_sum[j].unsqueeze(0) for j in mask_index[i]],
                                dim = -1
            ))
        '''
        outs = torch.cat((cls, out_vecs), dim = 1)
        outs = outs.to(torch.device("cuda"))
        pooled_output = self.dropout(outs)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

```






### <-
