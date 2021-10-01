## 누구 것을 볼 것인가?
- [김준홍](#김준홍)
- [안영진](#안영진)
- [최성욱](#최성욱)

### 김준홍

- preprocessing 부분만 손봤음.

```python
import pickle as pickle
import os
import pandas as pd
import torch
import re


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def __size__(self):
        return


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    sentences = []
    for i, j, s in zip(dataset["subject_entity"], dataset["object_entity"], dataset["sentence"]):
        # 기존 코드로는 , 가 들어있는 entity의 경우 제대로 분리해주지 못함.
        # 출처: https://stages.ai/competitions/75/discussion/talk/post/641 , 댓글 김민식님
        i = i[i.find("word") + 8 : i.find("start_idx") - 4]
        j = j[j.find("word") + 8 : j.find("start_idx") - 4]
        s = re.sub(r"[“ ”]", '"', s)

        subject_entity.append(i)
        object_entity.append(j)
        sentences.append(s)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": sentences,
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + tokenizer.sep_token + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        # return_token_type_ids=False,
    )

    return tokenized_sentences

```

### 안영진

```python
import torch
from torch.utils.data import DataLoader, Dataset
class CustomDataset(Dataset):
    """
    Yields dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'label', 'subject_entity', 'object_entity'])
    """
    def __init__(
        self,
        df_data:pd.DataFrame,
        tokenizer:AutoTokenizer,
        max_token_length:int,
        stopwords:list):
        """ tokenized input and label """
        self.label = torch.tensor(label_to_num(label=df_data['label'].values))
        self.sentence = df_data['sentence']
        self.subject_entity = df_data['subject_entity']
        self.object_entity = df_data['object_entity']
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.stopwords = stopwords

    def __getitem__(self, index:int):
        """
        Get item in forms of dictionary based on index number.

        ### Side note for encode_plus: https://stackoverflow.com/a/61732210/8380469
        `encode_plus` will:
        (1) Tokenize the sentence.
        (2) Prepend the `[CLS]` token to the start.
        (3) Append the `[SEP]` token to the end.
        (4) Map tokens to their IDs.
        (5) Pad or truncate the sentence to `max_length`
        (6) Create attention masks for [PAD] tokens.
        """
        item_sentence = self.sentence[index]
        # tokenize sentence
        encoded_dict = \
            self.tokenizer.encode_plus(
            item_sentence, # Sentence to encode.
            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
            max_length = self.max_token_length, # Pad & truncate all sentences.
            pad_to_max_length = True,
            return_attention_mask = True,   # Construct attn. masks.
            return_tensors = 'pt',     # Return pytorch tensors.
            truncation=True
        )

        # make items' dictionary
        # encoded_dict["attention_mask"] = encoded_dict["attention_mask"].squeeze(-1)
        encoded_dict["label"] = self.label[index]
        s = self.subject_entity[index]
        o = self.object_entity[index]

        return encoded_dict

    def __len__(self):
        """ return length of dataset """
        return len(self.tokenized_input)
```

### 최성욱
```python
def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128, # default :256
        add_special_tokens=True,
        )
    return tokenized_sentences
```