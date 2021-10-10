import torch
import pickle
from ast import literal_eval
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
        sub_dict = literal_eval(i)
        obj_dict = literal_eval(j)

        sub_start = int(sub_dict['start_idx'])
        sub_end = int(sub_dict['end_idx'])
        sub_type = sub_dict['type']

        obj_start = int(obj_dict['start_idx'])
        obj_end = int(obj_dict['end_idx'])
        obj_type = obj_dict['type']

        subject_entity.append([sub_start, sub_end, sub_type])
        object_entity.append([obj_start, obj_end, obj_type])
    out_dataset = pd.DataFrame({'id': dataset['id'],
                                'sentence': dataset['sentence'],
                                'subject_entity': subject_entity,
                                'object_entity': object_entity,
                                'label': dataset['label'], })
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
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
        "POH": "기타"
    }
    sentences = []
    e01, e02, sent = dataset['subject_entity'], dataset['object_entity'], dataset['sentence']
    subject_start, subject_end, sub_type = e01
    object_start, object_end, obj_type = e02
    subj = sent[e01[0]: e01[1] + 1]
    obj = sent[e02[0]: e02[1] + 1]
    if subject_start < object_start:
        sent_ = sent[:subject_start] + f'@*{type_dict[sub_type]}*' + subj + '@' + \
                    sent[subject_end + 1:object_start] + f'&^{type_dict[obj_type]}^' \
                    + obj + '&' + sent[object_end + 1:]
        ss = 1 + len(tokenizer.tokenize(sent[:subject_start]))
        se = ss + 4 + len(tokenizer.tokenize(subj))
        es = 1 + se + len(tokenizer.tokenize(sent[subject_end + 1:object_start]))
        ee = es + 4 + len(tokenizer.tokenize(obj))
    else:
        sent_ = sent[:object_start] + f'&^{type_dict[obj_type]}^' + obj + '&' + \
                sent[object_end + 1:subject_start] + f'@*{type_dict[sub_type]}*' + subj + '@' + \
                sent[subject_end + 1:]
        es = 1 + len(tokenizer.tokenize(sent[:object_start]))
        ee = es + 4 + len(tokenizer.tokenize(obj))
        ss = 1 + ee + len(tokenizer.tokenize(sent[object_end + 1:subject_start]))
        se = ss + 4 + len(tokenizer.tokenize(subj))
    sentences.append(sent_)
    max_length = 256
    senttokens = tokenizer.tokenize(sent_)[:max_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(senttokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    return input_ids, ss, se, es, ee


def collate_fn(batch):
    '''
    Retrieving the input_ids, input_mask, labels, subject and object start positions
    for IB model
    '''
    max_len = 256
    input_ids = [f["input_ids"] + [1] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    se = [f['se'] for f in batch]
    es = [f["es"] for f in batch]
    ee = [f['ee'] for f in batch]
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
    with open('./code/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


def processor(tokenizer, dataset, train_mode):
    '''
    train_dataset = processor(tokenizer, train_df))
    --> train_dataloader = Dataloader(train_dataset, batch_size = ...)
    '''
    features = []
    labels = dataset['label'].values
    if train_mode:
        labels = label_to_num(dataset['label'].values)
    for i in range(len(dataset)):
        input_ids, new_ss, new_se, new_es, new_ee = tokenized_dataset(dataset.iloc[i], tokenizer)
        label = labels[i]
        feature = {
            'input_ids' : input_ids,
            'labels' : label,
            'ss': new_ss,
            'se': new_se,
            'es' : new_es,
            'ee' : new_ee,
        }
        features.append(feature)
    return features


def split_df(df, kfold_n):
    kfold = StratifiedKFold(n_splits = kfold_n)
    X = df['sentence'].values
    y = df['label'].values
    datas = []
    for i, (train_index, valid_index) in enumerate(kfold.split(X,y)):
        train_df = df.iloc[train_index].copy().reset_index(drop=True)
        valid_df = df.iloc[valid_index].copy().reset_index(drop=True)

        datas.append((train_df, valid_df))
    return datas