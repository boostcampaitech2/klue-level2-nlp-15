```python
""" Set configuration as dictionary format """

import wandb
from datetime import datetime
from easydict import EasyDict

# login wandb and get today's date until hour and minute

wandb.login()
today = datetime.now().strftime("%m%d\_%H:%M")

# Debug set to true in order to debug high-layer code.

# CFG Configuration

CFG = wandb.config # wandb.config provides functionality of easydict.EasyDict
CFG.DEBUG = True

# Dataset Config as constants

CFG.num_labels = 30
CFG.num_multi_labels = 29
CFG.num_workers = 0
CFG.split_ratio = 0 # not going to use validation/test set
CFG.batch_size = 32

# Train configuration

CFG.user*name = "snoop2head"
CFG.file_base_name = f"{CFG.user_name}*{today}"
CFG.model_name = "klue/roberta-base" # https://huggingface.co/klue/roberta-base
CFG.num_folds = 5 # 5 Fold as default
CFG.num_epochs = 5 # loss is increasing after 5 epochs
CFG.max_token_length = 128 # refer to EDA where Q3 is 119, there are only 460 sentences out of 32k train set
CFG.stopwords = []
CFG.learning_rate = 5e-5
CFG.weight_decay = 1e-2 # https://paperswithcode.com/method/weight-decay
CFG.input_size = 768
CFG.output_size = 768
CFG.num_rnn_layers = 3
CFG.dropout_rate = 0.0

# training steps configurations

CFG.save_steps = 500
CFG.early_stopping_patience = 5
CFG.warmup_steps = 500
CFG.logging_steps = 100
CFG.evaluation_strategy = 'epoch'
CFG.evaluation_steps = 500

# Directory configuration

CFG.result_dir = os.path.join(ROOT_PATH, "results")
CFG.saved_model_dir = os.path.join(ROOT_PATH, "best_models")
CFG.logging_dir = os.path.join(ROOT_PATH, "logs")

# file configuration

CFG.result_file = os.path.join(CFG.result_dir, f"{CFG.file_base_name}.csv")
CFG.saved_model_file = os.path.join(CFG.saved_model_dir, f"{CFG.file_base_name}.pth")
CFG.logging_file = os.path.join(CFG.logging_dir, f"{CFG.file_base_name}.log")
CFG.train_file_path = TRAIN_FILE_PATH
CFG.test_file_path = TEST_FILE_PATH
CFG.sample_submission_file_path = SAMPLE_SUBMISSION_PATH

# Other configurations

CFG.seed = 2021
CFG.load_best_model_at_end = True
```
