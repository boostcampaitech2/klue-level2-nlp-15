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
