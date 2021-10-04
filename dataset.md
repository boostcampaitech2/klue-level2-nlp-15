## 누구 것을 볼 것인가?
- [김준홍](#김준홍)
- [안영진](#안영진)
- [최성욱](#최성욱)
- [전재영](#전재영)
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
        s = re.sub(r"[“”]", '"', s)

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
def pull_out_dictionary(df_input: pd.DataFrame):

    df = df_input.copy()

    # assign subject_entity and object_entity column values type as dictionary
    df['subject_entity'] = df['subject_entity'].apply(lambda x: eval(x))
    df['object_entity'] = df['object_entity'].apply(lambda x: eval(x))

    # parse item inside of subject_entity and object_entity's dictionary values as columns of dataframe
    # word, start_idx, end_idx, type as new columns 
    df = df.assign(
        # subject_entity
        subject_word=lambda x: x['subject_entity'].apply(lambda x: x['word']),
        subject_start_idx=lambda x: x['subject_entity'].apply(lambda x: x['start_idx']),
        subject_end_idx=lambda x: x['subject_entity'].apply(lambda x: x['end_idx']),
        subject_type=lambda x: x['subject_entity'].apply(lambda x: x['type']),
        
        # object_entity
        object_word=lambda x: x['object_entity'].apply(lambda x: x['word']),
        object_start_idx=lambda x: x['object_entity'].apply(lambda x: x['start_idx']),
        object_end_idx=lambda x: x['object_entity'].apply(lambda x: x['end_idx']),
        object_type=lambda x: x['object_entity'].apply(lambda x: x['type']),
    )

    # drop subject_entity and object_entity column
    df = df.drop(['subject_entity', 'object_entity'], axis=1)

    return df

df_train = pull_out_dictionary(df_train)
df_test = pull_out_dictionary(df_test)
```

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

### 전재영

```python
<sub>단어</sub>, <ob>단어</ob> 로 만들어주는 함수도 추가했습니다. -> 이때는 토크나이저에 이 토큰들을 추가하고 모델의 임베딩 사이즈를 바꿔주세요
```


```python
import pickle as pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, random_split


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]).clone().detach()
            for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def split_dataset(self, ratio=0.2):
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    dataset["sentence"] = dataset["sentence"].map(lambda x: x.replace("”", '"'))
    dataset["sentence"] = dataset["sentence"].map(lambda x: x.replace('" "', ""))
    dataset["sentence"] = dataset["sentence"].map(lambda x: x.replace("' '", ""))
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_more_data(dataset_dir, ratio=0.2):
    """ return train_set, test_set, if you want you can get augmentation in tran_set"""
    pd_dataset = pd.read_csv(dataset_dir)

    valid_len = int(len(pd_dataset) * 0.2)
    train_set, dev_set = (
        pd_dataset.loc[: len(pd_dataset) - valid_len],
        pd_dataset.loc[len(pd_dataset) - valid_len :],
    )

    train_set = make_more_data(train_set)

    train_set = preprocessing_dataset(train_set)
    dev_set = preprocessing_dataset(dev_set)
    return train_set, dev_set


def make_more_data(dataset):
    # if with changing label key to value, make more data
    change_dict = {
        "org:member_of": "org:members",
        "org:members": "org:member_of",
        "per:spouse": "per:spouse",
        "per:colleagues": "per:colleagues",
        "per:parents": "per:children",
        "per:children": "per:parents",
        "per:other_family": "per:other_family",
        "per:siblings": "per:siblings",
        "org:top_members/employees": "per:employee_of",
        "per:employee_of": "org:top_members/employees",
        "org:alternate_names": "org:alternate_names",
        "per:alternate_names": "per:alternate_names",
    }
    x = len(dataset)
    for i in range(x):
        if dataset.loc[i]["label"] in change_dict:
            dataset = dataset.append(
                {
                    "sentence": dataset.loc[i]["sentence"],
                    "subject_entity": dataset.loc[i]["object_entity"],
                    "object_entity": dataset.loc[i]["subject_entity"],
                    "label": change_dict[dataset.loc[i]["label"]],
                    "source": dataset.loc[i]["source"],
                },
                ignore_index=True,
            )
    return dataset


def preprocessing_datasets(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        subject_entity.append(i["word"])
        object_entity.append(j["word"])
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def my_tokenized_dataset(dataset, tokenizer):
    """ tokenized dataset, but unlike tokenized_dataset reslut is only original setence, not subject word <sep> object word <sep> original sentence"""
    tokenized_sentences = tokenizer(
        list(dataset["sentence"]),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=514,
        add_special_tokens=True,
    )
    return tokenized_sentences


def tokenize(sentence, subject_entity, object_entity):
    """ replace subject word to '<sub>word</sub>
        replace subject word to '<ob>word</ob>
     """
    sentence = (
        sentence[: subject_entity["start_idx"]]
        + "<sub>"
        + subject_entity["word"]
        + "</sub>"
        + sentence[subject_entity["end_idx"] + 1 :]
    )
    sentence = sentence.replace(
        object_entity["word"], "<ob>" + object_entity["word"] + "</ob>"
    )
    return sentence


def load_my_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    pd_dataset["subject_entity"] = pd_dataset["subject_entity"].map(lambda x: eval(x))
    pd_dataset["object_entity"] = pd_dataset["object_entity"].map(lambda x: eval(x))
    pd_dataset["sentence"] = pd_dataset.apply(
        lambda x: tokenize(x["sentence"], x["subject_entity"], x["object_entity"]),
        axis=1,
    )
    # print(pd_dataset['sentence'][0])
    dataset = preprocessing_datasets(pd_dataset)

    return dataset



```


