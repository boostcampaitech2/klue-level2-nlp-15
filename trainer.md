## Tariner를 보고싶다면?! 이름을 끌릭하세요~!
- [안영진](#안영진)
- [최성욱](#최성욱)
- [전재영](#전재영)

### 안영진

```python
# set device using torch
if torch.cuda.is_available() and CFG.DEBUG == False:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Device: ", device)
```

---

```python
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
```

```python
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Subset
from adamp import AdamP


# define train dataset
train = CustomizedDataset(df_train, tokenizer, CFG.max_token_length, CFG.stopwords)
valid = copy.deepcopy(train)

# use stratified k-fold cross validation -> train.py
stratified_kfold = StratifiedKFold(n_splits=CFG.num_folds, shuffle=False, random_state=None)
k_index = 0
for train_index, valid_index in stratified_kfold.split(df_train, df_train['label']):

    # Define KFolds
    k_index += 1
    print(f'####### K-FOLD :: {k_index}th')
    #slack_noti.noti(f'####### K-FOLD :: {k_idx}th')
    train_set = Subset(train, train_index)
    valid_set = Subset(valid, valid_index)
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    # Define Model
    model = CustomModel(
        CFG.model_name,
        CFG.num_labels,
        CFG.num_multi_labels,
        CFG.input_size,
        CFG.output_size,
        CFG.num_rnn_layers,
        CFG.dropout_rate
        )
    model.to(device)

    # define optimizer
    optimizer = AdamP(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999), weight_decay=CFG.weight_decay)

    # define scheduler: https://pytorch.org/docs/stable/optim.html
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)

    # define criterion
    criterion = FocalLoss(gamma = 5)
    # criterion = AngularPenaltySMLoss(in_features, out_features, loss_type='arcface') # loss_type in ['arcface', 'sphereface', 'cosface']

    # initialize average meters
    train_loss_meter = Metrics()
    train_acc_meter = Metrics()
    valid_loss_meter = Metrics()
    valid_acc_meter = Metrics()

    # initialize accuracy
    best_acc = 0.0
    steps = 0

    # train
    for epoch in range(1, CFG.num_epochs+1):
        # change the model's mode into training mode
        model.train()
        for i, batch in enumerate(train_loader):
            # clear gradients
            optimizer.zero_grad()

            # forward
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['token_type_ids'] = batch['token_type_ids'].to(device)
            batch['label'] = batch['label'].to(device)
            batch['subject_entity'] = batch['subject_entity'].to(device)
            batch['object_entity'] = batch['object_entity'].to(device)

            # get output
            output = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['token_type_ids'],
                batch['subject_entity'],
                batch['object_entity']
                )

            # get loss
            loss = criterion(output, batch['label'])

            # backward
            loss.backward()
            optimizer.step()
            scheduler.step()

            # update meters
            acc = accuracy_score(batch['label'].cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy())
            train_loss_meter.update(loss.item(), n=batch['label'].size(0))
            train_acc_meter.update(acc, n=batch['label'].size(0))

            # print log
            if steps % 100 == 0:
                print(f'[{steps}/{len(train_loader)}]\t'
                    f'train_loss: {train_loss_meter.avg:.4f}\t'
                    f'train_acc: {train_acc_meter.avg:.4f}')

                # validation accuracy calculation
                model.eval()
                with torch.no_grad():
                    for j, batch in enumerate(valid_loader):
                        batch['input_ids'] = batch['input_ids'].to(device)
                        batch['attention_mask'] = batch['attention_mask'].to(device)
                        batch['token_type_ids'] = batch['token_type_ids'].to(device)
                        batch['label'] = batch['label'].to(device)
                        batch['subject_entity'] = batch['subject_entity'].to(device)
                        batch['object_entity'] = batch['object_entity'].to(device)

                        # get output
                        output = model(
                            batch['input_ids'],
                            batch['attention_mask'],
                            batch['token_type_ids'],
                            batch['subject_entity'],
                            batch['object_entity']
                            )

                        # get loss
                        loss = criterion(output, batch['label'])

                        # update meters
                        acc = accuracy_score(batch['label'].cpu().detach().numpy(), output.argmax(dim=1).cpu().detach().numpy())
                        valid_loss_meter.update(loss.item(), n=batch['label'].size(0))
                        valid_acc_meter.update(acc, n=batch['label'].size(0))

                        # print log
                        if steps % 100 == 0:
                            print(f'[{steps}/{len(train_loader)}]\t'
                                f'valid_loss: {valid_loss_meter.avg:.4f}\t'
                                f'valid_acc: {valid_acc_meter.avg:.4f}')

                # save model
                if valid_acc_meter.avg > best_acc:
                    best_acc = valid_acc_meter.avg
                    torch.save(model.state_dict(), f'{CFG.model_name}_{k_index}_{CFG.num_epochs}_epochs.pth')
                    print(f'[{k_index}/{CFG.k_fold}]\t'
                        f'best_acc: {best_acc:.4f}')

                # reset meters
                train_loss_meter.reset()
                train_acc_meter.reset()
                valid_loss_meter.reset()
                valid_acc_meter.reset()

            steps += 1

```

### 최성욱
#### Stratified-KFold 첫번째 Try (with Inference)
- Stratified-KFold를 기존의 방식으로 한 이유
    - HuggingFace의 Trainer를 사용할 경우, 최종 Average score를 어떻게 출력해야하는지 모름.
    - end-to-end로 inference까지 진행된다. 최종 output만 결과 추출
        - 안좋다! -> 중간에 저장하는 것도 중요하고, 분업도 힘듬 -> Seconde Try 진행
- /# of fold : 5 / epoch : 1 / 
- Micro_f1_score : 64.046 / Auprc : 68.871 

- train
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.optim import AdamW

stf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 1
patience = 2
results = {}
oof_pred = None
check = []

for fold, (train_idx, dev_idx) in enumerate(stf.split(train_dataset, list(train_dataset['label']))) :
    best_val_loss = 9999
    counter = 0

    # -- raw dataset
    train = train_dataset.iloc[train_idx]
    dev = train_dataset.iloc[dev_idx]

    train_label = label_to_num(train['label'].values)
    dev_label = label_to_num(dev['label'].values)

    # -- tokenized
    tokenized_train = tokenized_dataset(train, tokenizer)
    tokenized_dev = tokenized_dataset(dev, tokenizer)

    # -- dataset
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    # -- model
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.to(device)
    
    # -- dataloader
    RE_train_loader = DataLoader(RE_train_dataset, batch_size = 16, shuffle = True)
    RE_dev_loader = DataLoader(RE_dev_dataset, batch_size = 16, shuffle = False)

    # -- optimzer
    optim = AdamW(model.parameters(), lr = 5e-5)

    # -- train
    for epoch in range(1) :
        model.train()

        loss_value = 0
        acc = 0
        micro_f1_value = 0
        auprc = 0

        n_iter = 0
        for idx, batch in enumerate(RE_train_loader) :
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optim.zero_grad()
            outputs = model(input_ids, 
                            attention_mask = attention_mask,
                            token_type_ids = token_type_ids,
                            labels = labels)
            
            loss = outputs[0]
            loss.backward()
            optim.step()

            preds = torch.argmax(outputs[1], dim = -1)
            probs = outputs[1]

            loss_value += loss.item()
            if (idx + 1) % 100 == 0 :
                train_loss = loss_value / 100
                print(
                        f"Epoch[{epoch}/{epochs}]({idx + 1}/{len(RE_train_loader)}) || "
                        f"training loss {train_loss:4.4}"
                    )
                loss_value = 0

        with torch.no_grad() :
            print('Calculating validation results ...')
            model.eval()

            val_loss_items = []

            for i, val_batch in enumerate(RE_dev_loader) :
                input_ids = val_batch['input_ids'].to(device)
                attention_mask = val_batch['attention_mask'].to(device)
                token_type_ids = val_batch['token_type_ids'].to(device)
                labels = val_batch['labels'].to(device)

                outputs = model(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = labels)
                loss = outputs[0]


                loss_item = loss.item()
                val_loss_items.append(loss_item)

            val_loss = np.sum(val_loss_items) / len(RE_dev_loader)
            print(f'valdiation loss {val_loss:4.4}')

            best_val_loss = min(best_val_loss, val_loss)
            if val_loss < best_val_loss :
                print(f'New best model for val accuracy : {val_loss:4.2%}! saving the best model..')
                model.save_pretrained(f'./best_model/kfold/{fold + 1}/')
                best_val_loss = val_loss
                counter = 0
            else :
                counter += 1
            
            if counter > patience :
                print('Early Stopping....')
                counter = 0
                break
    results[fold] = [best_val_loss]

    # Inference
    test_dataset_dir = "../dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset ,test_label) 

    dataloader = DataLoader(Re_test_dataset, batch_size=16, shuffle=False)
    model.eval()

    output_logits = []
    output_pred = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
                )
        logits = outputs[0] # evaluation mode에서는 0번이 logit
        prob = F.softmax(logits, dim = -1).detach().cpu().numpy()
        # prob과 logits을 np.argmax()를 태우면 결과는 똑같다.
        # 차이는 prob은 np.array, logits은 list
        output_pred.append(prob)

    final_prob = np.concatenate(output_pred, axis = 0)

    # fold ensemble
    if oof_pred is None :
        oof_pred = final_prob / 5
    else :
        oof_pred += final_prob / 5
```

- inference
```python
average_loss = 0.
for k, v in results.items() :
    average_loss += v[0]
print(f'Loss Average: {average_loss/5:4.4}')

result = np.argmax(oof_pred, axis = -1)
pred_answer = num_to_label(result)
output_prob = oof_pred.tolist()

output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 
```

#### Stratified-KFold Second Try (with Inference)
- HuggingFace Trainer를 사용하여 model 저장한 후 불러서 inference하는 방식
- train -> model_save -> model_load -> inference
- batch_size : 32 / epoch : 2 / 
- micro_f1_score : 66.296 / Auprc : 70.888

- train
```python
dataset = load_data("../dataset/train/train.csv")

training_args = TrainingArguments(
  output_dir='./results',          # output directory
  save_total_limit=5,              # number of total save model.
  save_steps=500,                   # model saving step.
  num_train_epochs=2,              # total number of training epochs
  learning_rate=5e-5,               # learning_rate
  per_device_train_batch_size=32,  # batch size per device during training
  per_device_eval_batch_size=32,   # batch size for evaluation
  warmup_steps=500,                # number of warmup steps for learning rate scheduler
  weight_decay=0.01,               # strength of weight decay
  logging_dir='./logs',            # directory for storing logs
  logging_steps=100,              # log saving step.
  evaluation_strategy='steps', # evaluation strategy to adopt during training
                              # `no`: No evaluation during training.
                              # `steps`: Evaluate every `eval_steps`.
                              # `epoch`: Evaluate every end of epoch.
  eval_steps = 500,            # evaluation step.
  load_best_model_at_end = True 
)

# 본 제출(max_length = 128)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

models = []
stf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed_everything(42))
for fold, (train_idx, dev_idx) in enumerate(stf.split(dataset, list(dataset['label']))) :
    print('Fold {}'.format(fold + 1))
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config = model_config)
    model.to(device)

    train_dataset = dataset.iloc[train_idx]
    dev_dataset = dataset.iloc[dev_idx]

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
    )
    trainer.train()
    models.append(model)
```

- model save & load
```python
def makedirs(path) :
    try :
        os.makedirs(path)
    except OSError :
        if not os.path.isdir(path) :
            raise

for i, model in enumerate(models) :
    makedirs(f'./best_model/kfold/fold_{i}/')
    model.save_pretrained(f'./best_model/kfold/fold_{i}/')
```

- inference
```python
test_dataset_dir = "../dataset/test/test_data.csv"
test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
Re_test_dataset = RE_Dataset(test_dataset ,test_label) 

dataloader = DataLoader(Re_test_dataset, batch_size=32, shuffle=False)

oof_pred = None
for i in range(5) :
    model_name = '/opt/ml/code/best_model/kfold/fold_{}'.format(i)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    output_pred = []
    for i, data in enumerate(tqdm(dataloader)) :
        with torch.no_grad() :
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
                )
        logits = outputs[0]
        prob = F.softmax(logits, dim = -1).detach().cpu().numpy()
        output_pred.append(prob)
    final_prob = np.concatenate(output_pred, axis = 0)

    if oof_pred is None :
        oof_pred = final_prob / 5
    else :
        oof_pred += final_prob / 5

result = np.argmax(oof_pred, axis = -1)
pred_answer = num_to_label(result)
output_prob = oof_pred.tolist()

output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
```


### 전재영
#### two model inference 입니다(결과가 좋지는 않네요 ㅠ)
```python
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn

""" model1: 관계가 있는 데이터(1-29의 데이터)로 학습한 모델, 데이터의 관계를 뽑아냄, 1-29의 결과를 내뱉지만 라벨은 30개가 있습니다.
    model2: train set에 대해 관계가 있는지 없는지 두 개의 라벨로 학습한 라벨
모든 데이터에 대하여 model 1으로  29개의 라벨을 뽑은 뒤에, model 2가 no_relation으로 예측한 정답을 no_relation으로 바꿔주는 과정을 거칩니다.

def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                # token_type_ids=data["token_type_ids"].to(device),
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)

    test_label = list(map(int, test_dataset["label"].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset["id"], tokenized_test, test_label

def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    Tokenizer_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model , model1은 29개의 관계를 뽑는 모델, model2는 no_relation과 relation을 뽑는 모델
    # MODEL_NAME = args.model_dir1  # model dir.
    model1 = AutoModelForSequenceClassification.from_pretrained(args.model_dir1)
    model2 = AutoModelForSequenceClassification.from_pretrained(args.model_dir2)
    model1.to(device)  
    model2.to(device) 

    ## load test datset
    test_dataset_dir = "../dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(
        model1, Re_test_dataset, device
    )  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    ### 위에까지는 모델을 두개 불러오는 것 뺴고는 inference.py와 같습니다
 
    with torch.no_grad():
        m = nn.Softmax(dim=1)
        for idx in tqdm(range(len(Re_test_dataset))):
            data = {i: v.to(device).unsqueeze(0) for i, v in Re_test_dataset[idx].items()}
            out = model2(input_ids = data['input_ids'],attention_mask = data['attention_mask'])
            logit = out['logits']
            logit = m(logit)
            predicted = torch.argmax(logit, -1)
            if predicted == 0:  # 예측이 0이면
                output["pred_label"][idx] = 'no_relation'
                #확률변화- 기존의 확률에서 label 0의 확률은 model 2의 0의 확률로 바꾸고 전체의 합이 1이 되도록 정규화
                output["probs"][idx][0] = logit[0][0].item()
                output["probs"][idx] = torch.tensor(output["probs"][idx]) / torch.sum(torch.tensor(output["probs"][idx]))
                output["probs"][idx] = output["probs"][idx].tolist()
    output.to_csv(
        "./prediction/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir1", type=str, default="./best_model1")
    parser.add_argument("--model_dir2", type=str, default="./best_model2")
    args = parser.parse_args()
    print(args)
    main(args)

```
