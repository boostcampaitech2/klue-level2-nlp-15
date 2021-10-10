import torch
import pandas as pd
from collections import defaultdict
import random
import pickle

train_dir = "../dataset/train/train_fixed.csv"


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
