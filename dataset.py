# import python innate functions
import random
import pickle
from ast import literal_eval
from collections import defaultdict

# import dataset wrangler
import numpy as np
import pandas as pd

# import machine learning modules
from sklearn.model_selection import StratifiedKFold

# import torch and its applications
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# import from huggingface transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM


# import third party modules
import yaml

DATA_CFG = {}
IB_CFG = {}
RBERT_CFG = {}
CONCAT_CFG = {}

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)

DATA_CFG = SAVED_CFG["data"]
IB_CFG = SAVED_CFG["IB"]
RBERT_CFG = SAVED_CFG["RBERT"]
CONCAT_CFG = SAVED_CFG["Concat"]

###############################################################################

## dataset for Improved Baseline models


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        sub_dict = literal_eval(i)
        obj_dict = literal_eval(j)

        sub_start = int(sub_dict["start_idx"])
        sub_end = int(sub_dict["end_idx"])
        sub_type = sub_dict["type"]

        obj_start = int(obj_dict["start_idx"])
        obj_end = int(obj_dict["end_idx"])
        obj_type = obj_dict["type"]

        subject_entity.append([sub_start, sub_end, sub_type])
        object_entity.append([obj_start, obj_end, obj_type])
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


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)
    return dataset


def tokenized_dataset(dataset, tokenizer):
    """
    Inserting typed entity markers to each sentences
    subject: @*type*subject word@ (e.g.  김현수 -> @*사람*김현수@)
    object: #^type^object word# (e.g. #^지명^한국#)

    <<An Improved Baseline for Sentence-level Relation Extraction>>

    returns: input_ids, subject start & end positions, object start & end positions

    """

    type_dict = {
        "PER": "사람",
        "LOC": "지명",
        "ORG": "기관",
        "DAT": "날짜",
        "TIM": "시간",
        "DUR": "기간",
        "MNY": "통화",
        "PNT": "비율",
        "NOH": "수량",
        "POH": "기타",
    }
    sentences = []
    e01, e02, sent = (
        dataset["subject_entity"],
        dataset["object_entity"],
        dataset["sentence"],
    )
    subject_start, subject_end, sub_type = e01
    object_start, object_end, obj_type = e02
    subj = sent[e01[0] : e01[1] + 1]
    obj = sent[e02[0] : e02[1] + 1]
    if subject_start < object_start:
        sent_ = (
            sent[:subject_start]
            + f"@*{type_dict[sub_type]}*"
            + subj
            + "@"
            + sent[subject_end + 1 : object_start]
            + f"&^{type_dict[obj_type]}^"
            + obj
            + "&"
            + sent[object_end + 1 :]
        )
        ss = 1 + len(tokenizer.tokenize(sent[:subject_start]))
        se = ss + 4 + len(tokenizer.tokenize(subj))
        es = 1 + se + len(tokenizer.tokenize(sent[subject_end + 1 : object_start]))
        ee = es + 4 + len(tokenizer.tokenize(obj))
    else:
        sent_ = (
            sent[:object_start]
            + f"&^{type_dict[obj_type]}^"
            + obj
            + "&"
            + sent[object_end + 1 : subject_start]
            + f"@*{type_dict[sub_type]}*"
            + subj
            + "@"
            + sent[subject_end + 1 :]
        )
        es = 1 + len(tokenizer.tokenize(sent[:object_start]))
        ee = es + 4 + len(tokenizer.tokenize(obj))
        ss = 1 + ee + len(tokenizer.tokenize(sent[object_end + 1 : subject_start]))
        se = ss + 4 + len(tokenizer.tokenize(subj))
    sentences.append(sent_)
    max_length = 256
    senttokens = tokenizer.tokenize(sent_)[: max_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(senttokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    return input_ids, ss, se, es, ee


def collate_fn(batch):
    """
    Retrieving the input_ids, input_mask, labels, subject and object start positions
    for IB model
    """
    max_len = 256
    input_ids = [f["input_ids"] + [1] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [
        [1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"]))
        for f in batch
    ]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    se = [f["se"] for f in batch]
    es = [f["es"] for f in batch]
    ee = [f["ee"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    se = torch.tensor(se, dtype=torch.long)
    es = torch.tensor(es, dtype=torch.long)
    ee = torch.tensor(ee, dtype=torch.long)
    output = (input_ids, input_mask, labels, ss, se, es, ee)
    return output


def label_to_num(label):
    num_label = []
    with open("data/dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


def processor(tokenizer, dataset, train_mode):
    """
    train_dataset = processor(tokenizer, train_df))
    --> train_dataloader = Dataloader(train_dataset, batch_size = ...)
    """
    features = []
    labels = dataset["label"].values
    if train_mode:
        labels = label_to_num(dataset["label"].values)
    for i in range(len(dataset)):
        input_ids, new_ss, new_se, new_es, new_ee = tokenized_dataset(
            dataset.iloc[i], tokenizer
        )
        label = labels[i]
        feature = {
            "input_ids": input_ids,
            "labels": label,
            "ss": new_ss,
            "se": new_se,
            "es": new_es,
            "ee": new_ee,
        }
        features.append(feature)
    return features


def split_df(df, kfold_n):
    kfold = StratifiedKFold(n_splits=kfold_n)
    X = df["sentence"].values
    y = df["label"].values
    datas = []
    for i, (train_index, valid_index) in enumerate(kfold.split(X, y)):
        train_df = df.iloc[train_index].copy().reset_index(drop=True)
        valid_df = df.iloc[valid_index].copy().reset_index(drop=True)

        datas.append((train_df, valid_df))
    return datas


###############################################################################


class Data:
    def __init__(self, sent, se, oe, label):
        self.sentence = sent
        self.subject_entity = eval(se)
        self.object_entity = eval(oe)
        self.label = label

    def __repr__(self):
        sword = self.subject_entity["word"]
        oword = self.object_entity["word"]
        return self.sentence.replace(sword, f"[SUB]{sword}[/SUB]").replace(
            oword, f"[OBJ]{oword}[/OBJ]"
        )


def add_entity_token(data):
    sub_start_idx, sub_end_idx = (
        data.subject_entity["start_idx"],
        data.subject_entity["end_idx"],
    )
    obj_start_idx, obj_end_idx = (
        data.object_entity["start_idx"],
        data.object_entity["end_idx"],
    )

    sub_type = data.subject_entity["type"]
    obj_type = data.object_entity["type"]

    s = data.sentence

    if sub_start_idx < obj_start_idx:
        res = [
            s[:sub_start_idx],
            f"[SUB:{sub_type}]" + s[sub_start_idx : sub_end_idx + 1] + "[/SUB]",
            s[sub_end_idx + 1 : obj_start_idx],
            f"[OBJ:{obj_type}]" + s[obj_start_idx : obj_end_idx + 1] + "[/OBJ]",
            s[obj_end_idx + 1 :],
        ]
    else:
        res = [
            s[:obj_start_idx],
            f"[OBJ:{obj_type}]" + s[obj_start_idx : obj_end_idx + 1] + "[/OBJ]",
            s[obj_end_idx + 1 : sub_start_idx],
            f"[SUB:{sub_type}]" + s[sub_start_idx : sub_end_idx + 1] + "[/SUB]",
            s[sub_end_idx + 1 :],
        ]

    return "".join(res)


def split_dataset(ratio, train_dir):
    df = pd.read_csv(train_dir)
    label2data = defaultdict(list)

    for item in df.itertuples():
        data = Data(item.sentence, item.subject_entity, item.object_entity, item.label)
        label2data[item.label].append(data)

    train_dataset = []
    valid_dataset = []

    for label, data_list in label2data.items():
        random.shuffle(data_list)
        for i, data in enumerate(data_list):
            if i < len(data_list) * ratio:
                valid_dataset.append(data)
            else:
                train_dataset.append(data)

    random.shuffle(train_dataset)
    random.shuffle(valid_dataset)
    return train_dataset, valid_dataset


def generate_test_dataset(test_dir):
    test_dataset = []

    df = pd.read_csv(test_dir)
    for item in df.itertuples():
        data = Data(item.sentence, item.subject_entity, item.object_entity, item.label)
        test_dataset.append(data)

    return test_dataset


def tokenize_dataset(dataset, tokenizer, train=True, bi=False):
    concat_entities = []
    sentences = []
    labels = []
    entity_token_ids = []
    for data in dataset:
        sub_word_type = data.subject_entity["type"]
        obj_word_type = data.object_entity["type"]

        sub_entity_token_id = tokenizer.encode(f"[SUB:{sub_word_type}]")[1]
        obj_entity_token_id = tokenizer.encode(f"[OBJ:{obj_word_type}]")[1]

        token_added_sentence = add_entity_token(data)

        sentences.append(token_added_sentence)
        labels.append(data.label)
        entity_token_ids.append((sub_entity_token_id, obj_entity_token_id))

    tokenized_sentences = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    sub_token_indexes = []
    obj_token_indexes = []
    for tokens, (sub_token_id, obj_token_id) in zip(
        tokenized_sentences["input_ids"], entity_token_ids
    ):
        # print(tokens,sub_token_id,obj_token_id)
        try:
            sub_token_index = (tokens == sub_token_id).nonzero()[0].item()
            obj_token_index = (tokens == obj_token_id).nonzero()[0].item()
            sub_token_indexes.append(sub_token_index)
            obj_token_indexes.append(obj_token_index)
        except:
            print(
                tokenizer.decode(tokens), tokenizer.decode([sub_token_id, obj_token_id])
            )
            continue
    tokenized_sentences["sub_token_index"] = sub_token_indexes
    tokenized_sentences["obj_token_index"] = obj_token_indexes

    # tokenized_sentences['entity_token_ids'] = entity_token_ids

    if train == False:
        return tokenized_sentences

    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for i in range(len(labels)):
        if bi:
            labels[i] = 0 if labels[i] == "no_relation" else 1
        else:
            labels[i] = dict_label_to_num[labels[i]]

    return tokenized_sentences, labels


def load_train_dataset(train_dir, tokenizer):

    train_datalist, valid_datalist = split_dataset(ratio=0.2, train_dir=train_dir)

    train_tokenized, train_labels = tokenize_dataset(train_datalist, tokenizer)
    valid_tokenized, valid_labels = tokenize_dataset(valid_datalist, tokenizer)

    train_dataset = EntityRelationDataset(train_tokenized, train_labels)
    valid_dataset = EntityRelationDataset(valid_tokenized, valid_labels)

    return train_dataset, valid_dataset


class EntityRelationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_sentences, labels):
        self.tokenized_sentences = tokenized_sentences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: data[idx] for key, data in self.tokenized_sentences.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


###############################################################################


class RBERT_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, is_training: bool = True):

        # pandas.Dataframe dataset
        self.dataset = dataset
        self.sentence = self.dataset["sentence"]
        self.subject_entity = self.dataset["subject_entity"]
        self.object_entity = self.dataset["object_entity"]
        if is_training:
            # training mode
            self.train_label = label_to_num(self.dataset["label"].values)
        if not is_training:
            # test mode for submission
            self.train_label = self.dataset["label"].values
        self.label = torch.tensor(self.train_label)

        # tokenizer and etc
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        subject_entity = self.subject_entity[idx]
        object_entity = self.object_entity[idx]
        label = self.label[idx]

        # concat entity in the beginning
        concat_entity = subject_entity + "[SEP]" + object_entity

        # tokenize
        encoded_dict = self.tokenizer(
            concat_entity,
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=RBERT_CFG.max_token_length,
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )

        # RoBERTa's provided masks (do not include token_type_ids for RoBERTa)
        encoded_dict["input_ids"] = encoded_dict["input_ids"].squeeze(0)
        encoded_dict["attention_mask"] = encoded_dict["attention_mask"].squeeze(0)

        # add subject and object entity masks where masks notate where the entity is
        subject_entity_mask, object_entity_mask = self.add_entity_mask(
            encoded_dict, subject_entity, object_entity
        )
        encoded_dict["subject_mask"] = subject_entity_mask
        encoded_dict["object_mask"] = object_entity_mask

        # fill label
        encoded_dict["label"] = label
        return encoded_dict

    def __len__(self):
        return len(self.dataset)

    def add_entity_mask(self, encoded_dict, subject_entity, object_entity):
        """add entity token to input_ids"""
        # print("tokenized input ids: \n",encoded_dict['input_ids'])

        # initialize entity masks
        subject_entity_mask = np.zeros(RBERT_CFG.max_token_length, dtype=int)
        object_entity_mask = np.zeros(RBERT_CFG.max_token_length, dtype=int)

        # get token_id from encoding subject_entity and object_entity
        subject_entity_token_ids = self.tokenizer.encode(
            subject_entity, add_special_tokens=False
        )
        object_entity_token_ids = self.tokenizer.encode(
            object_entity, add_special_tokens=False
        )
        # print("entity token's input ids: ",subject_entity_token_ids, object_entity_token_ids)

        # get the length of subject_entity and object_entity
        subject_entity_length = len(subject_entity_token_ids)
        object_entity_length = len(object_entity_token_ids)

        # find coordinates of subject_entity_token_ids inside the encoded_dict["input_ids"]
        subject_coordinates = np.where(
            encoded_dict["input_ids"] == subject_entity_token_ids[0]
        )
        subject_coordinates = list(
            map(int, subject_coordinates[0])
        )  # change the subject_coordinates into int type
        for subject_index in subject_coordinates:
            subject_entity_mask[
                subject_index : subject_index + subject_entity_length
            ] = 1

        # find coordinates of object_entity_token_ids inside the encoded_dict["input_ids"]
        object_coordinates = np.where(
            encoded_dict["input_ids"] == object_entity_token_ids[0]
        )
        object_coordinates = list(
            map(int, object_coordinates[0])
        )  # change the object_coordinates into int type
        for object_index in object_coordinates:
            object_entity_mask[object_index : object_index + object_entity_length] = 1

        # print(subject_entity_mask)
        # print(object_entity_mask)

        return torch.Tensor(subject_entity_mask), torch.Tensor(object_entity_mask)


###############################################################################

## dataset for concat models


def pull_out_dictionary(df_input: pd.DataFrame):
    """pull out str `{}` values from the pandas dataframe and shape it as a new column"""

    df = df_input.copy()

    # assign subject_entity and object_entity column values type as dictionary
    df["subject_entity"] = df["subject_entity"].apply(lambda x: eval(x))
    df["object_entity"] = df["object_entity"].apply(lambda x: eval(x))

    # parse item inside of subject_entity and object_entity's dictionary values as columns of dataframe
    # word, start_idx, end_idx, type as new columns
    df = df.assign(
        # subject_entity
        subject_word=lambda x: x["subject_entity"].apply(lambda x: x["word"]),
        subject_start_idx=lambda x: x["subject_entity"].apply(lambda x: x["start_idx"]),
        subject_end_idx=lambda x: x["subject_entity"].apply(lambda x: x["end_idx"]),
        subject_type=lambda x: x["subject_entity"].apply(lambda x: x["type"]),
        # object_entity
        object_word=lambda x: x["object_entity"].apply(lambda x: x["word"]),
        object_start_idx=lambda x: x["object_entity"].apply(lambda x: x["start_idx"]),
        object_end_idx=lambda x: x["object_entity"].apply(lambda x: x["end_idx"]),
        object_type=lambda x: x["object_entity"].apply(lambda x: x["type"]),
    )

    # drop subject_entity and object_entity column
    df = df.drop(["subject_entity", "object_entity"], axis=1)

    return df


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset_concat(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

    dataset = pull_out_dictionary(dataset)

    # rename columns subject_word as subject_entity, object_word as object_entity
    dataset = dataset.rename(
        columns={"subject_word": "subject_entity", "object_word": "object_entity"}
    )

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": dataset["subject_entity"],
            "object_entity": dataset["object_entity"],
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data_concat(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset_concat(pd_dataset)

    return dataset


def tokenized_dataset_concat(dataset, tokenizer, max_token_length):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_token_length + 4,
        add_special_tokens=True,
        return_token_type_ids=False,
    )
    return tokenized_sentences
