from transformers import AutoTokenizer
from tqdm import tqdm
from dataloader.ib_dataset import *
from torch.utils.data import DataLoader
from utils.metrics import *
import torch.nn.functional as F
import os

def num_to_label(label):
    origin_label = []
    with open('./code/dict_label_to_num.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    new_dict = {value: key for key,value in dict_num_to_label.items()}
    for v in label:
        origin_label.append(new_dict[v])
    return origin_label


def inference_for_ib(model, test_features, device):
    dataloader = DataLoader(test_features, batch_size=16, shuffle=False, collate_fn = collate_fn)
    model.eval()
    output_pred = []
    output_prob = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'ss': batch[3].to(device),
                      'es': batch[5].to(device),
                      }
            outputs = model(**inputs)
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)
    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def load_test_dataset_for_ib(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_features = processor(tokenizer, test_dataset, train_mode=False)
    return test_dataset['id'], test_features

def inference_ib():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Tokenizer_NAME = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    for fold_num in range(5):
        MODEL_NAME = f"./re_finetuned/fold_ensemble/roberta_focal_adamp{fold_num}.pt'"
        model = torch.load(MODEL_NAME)
        model.parameters
        model.to(device)
        test_dataset_dir = "./dataset/test/test_data.csv"
        test_id, test_features = load_test_dataset_for_ib(test_dataset_dir, tokenizer)
        pred_answer, output_prob = inference_for_ib(model, test_features, device)
        pred_answer = num_to_label(pred_answer)
        output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })
        output.to_csv(f'./prediction/to_ensemble/output_p{fold_num}.csv', index=False)
    print('---- Finished making result files for each fold! ----')

    files = os.listdir('./prediction/to_ensemble')

    to_ensemble = [i for i in files if i.endswith(".csv")]
    total = []

    for i in tqdm(to_ensemble):
        df = pd.read_csv("./prediction/to_ensemble/" + i)
        tmp = [literal_eval(df.iloc[i]['probs']) for i in range(len(df))]
        total.append(tmp)

    avr_total = torch.sum(torch.tensor(total), dim=0) / 5

    result = np.argmax(avr_total, axis=-1)
    pred_answer = result.tolist()
    predsss = num_to_label(pred_answer)

    avr_total = avr_total.tolist()
    test_file = pd.read_csv("./dataset/test/test_data.csv")
    test_ids = test_file['id'].tolist()
    output = pd.DataFrame({"id": test_ids, "pred_label": predsss, "probs": avr_total}, )
    output.to_csv("./prediction/final_submission.csv", index=False)
    print('---- Finished creating Final ensembled file for all folds! ----')




def main():
    inference_ib()