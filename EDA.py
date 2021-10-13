import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import time
import datetime
import warnings
from transformers import AutoTokenizer
from dataset import *

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


df_train = load_data_concat(DATA_CFG["train_file_path"])
tokenizer = AutoTokenizer.from_pretrained(RBERT_CFG["pretrained_model_name"])
sentence = df_train["sentence"].tolist()

# get unk token's token id from tokenizer
unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# get encoded sentences from sentence
encoded_sentences = []
for item in sentence:
    tokenized_item = tokenizer(item, return_tensors="pt", add_special_tokens=True)
    encoded_sentences.append(tokenized_item["input_ids"][0])

# find unk_token_id from each encoded sentences
unk_token_id_list = []
known_token_id_list = []
for encoded_sentence in encoded_sentences:
    try:
        unk_token_id_list.append(list(encoded_sentence).index(unk_token_id))
    except:
        pass

unknown_count = len(unk_token_id_list)
# get vocab size of the tokenizer
no_sentences = len(encoded_sentences)

# print number of unknowns
print("Number of unknowns: ", unknown_count)
# print number of total sentences in the train dataset
print("Number of sentences: ", no_sentences)
