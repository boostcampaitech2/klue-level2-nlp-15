## Tariner를 보고싶다면?! 이름을 끌릭하세요~!
- [안영진](#안영진)
- [최성욱](#최성욱)

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
