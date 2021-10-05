[klue-level2-nlp-15](https://github.com/boostcampaitech2/klue-level2-nlp-15)

**dev branch에서 추가 & 개선할 점**

- [x] entity_1, entity_2, [CLS]에서 hidden state vector을 뽑아내서 concat하기  -> 세현

  - [ ] Improved Baseline -> 현수

- [ ] **Bidirectional RNN을 backbone 뒤에다가 붙여보기** 

  ```python
  class RoBERTa(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.MODEL_NAME = 'klue/roberta-large'
          self.bert_model = AutoModel.from_pretrained(self.MODEL_NAME)
          self.hidden_size = 1024
          self.num_labels = 30
          self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
          
          special_tokens_dict = {
              'additional_special_tokens':[
                  '[SUB:ORG]',
                  '[SUB:PER]',
                  '[/SUB]',
                  '[OBJ:DAT]',
                  '[OBJ:LOC]',
                  '[OBJ:NOH]',
                  '[OBJ:ORG]',
                  '[OBJ:PER]',
                  '[OBJ:POH]',
                  '[/OBJ]'
              ]
          }
  
          num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
          print("num_added_tokens:",num_added_tokens)
  
          self.bert_model.resize_token_embeddings(len(self.tokenizer))
          
          self.lstm = torch.nn.LSTM(input_size=~~~, hidden_size = ~~~,bidirectional=True)
          
          # classifier은 바꾸지 않고
          self.classifier = torch.nn.Sequential(
              torch.nn.Linear(3*self.hidden_size,self.hidden_size),
              torch.nn.Dropout(p=0.1, inplace=False),
              torch.nn.Linear(self.hidden_size,self.num_labels)
          )
    def forward(self,item):
      input_ids = item['input_ids']
      token_type_ids = item['token_type_ids']
      attention_mask = item['attention_mask']
      sub_token_index = item['sub_token_index']
      obj_token_index = item['obj_token_index']
      out = self.bert_model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
      h = out.last_hidden_state
      # output, _ = lstm(h,bidirectional=True)
      batch_size = h.shape[0]
  
  ```

  RBERT에서 나온 token이 문장 길이 10이었으면 10개가 있는데, 그걸 하나씩 넣으면서 10개 token에 대해서 sequence처리를 해야 할 것 같아요.

- [ ] Dropout(p=0)으로 Dropout 제거

- [ ] AdamP 사용하고 싶다.

- [ ] 현재 max_token_len = 256 -> collate_fn 만들어서 batch마다 max length 정해서 125보다 더 작은 length인 경우에는 더 짧게 해서 학습시키려고 했습니다 -> max_length를 구해서 하면 버리는 데이터도 없고 좋을것같아요 아예 데이터셋에서 batch를 묶어줄 때, 비슷한 token 개수(length)인 것들을 모아서 뽑는 방식도 있더라고요! 이걸 uniform length batching이라고 하는 것 같습니다. 어제 성욱님이 token length distribution이 class 별로도 유사하다는 걸 보여주신 덕분에 uniform length batching도 걱정없이 쓸 수 있을 것 같습니다 -> 영진

- [ ] Train set 전체를 입력하고 Random Mask (prob=0.1) 씌워서 Pretrain을 하고 싶다. 나만의 작은 KLUE-BERT만들기 -> 준홍, 재영, 영진 🤗

  - [ ] 실험할 때는 가벼운 걸(klue/bert-base)로 실험하고, 무거운 거(xlm-roberta-large, klue/roberta-large)로 pretrain시키기
  - [ ] ~~Train set하고 validation set하고 나눠서 Validation해야 의미가 있는 듯... (지금은 일단 validation으로 나눠놓는다) 근데 정답 주고 학습하는 건 아니니까 좀 애매... train함수 안에 MLM train할 생각이었는데. Train vs Val을 나눠놓은 다음에 그걸 train set에 MLM에 사용하고, MLM이 끝나면, finetuning을 할 생각이었음. random seed만 통일시켜놓는 것~~

- [ ] Pororo에서 NER로 표기한 42개 추가하기 

  ```python
  # models.py
  class RoBERTa(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.MODEL_NAME = 'klue/roberta-large'
          self.bert_model = AutoModel.from_pretrained(self.MODEL_NAME)
          self.hidden_size = 1024
          self.num_labels = 30
          self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
          
          special_tokens_dict = {
              'additional_special_tokens':[
                  '[SUB:ORG]',
                  '[SUB:PER]',
                  '[/SUB]',
                  '[OBJ:DAT]',
                  '[OBJ:LOC]',
                  '[OBJ:NOH]',
                  '[OBJ:ORG]',
                  '[OBJ:PER]',
                  '[OBJ:POH]',
                  '[/OBJ]'
              ]
          }
  
  # dataset.py에서 sub_type 부분을 바꾸면 됨
  def add_entity_token(data):
    """index로 하는 이유가 있다고!"""
      sub_start_idx, sub_end_idx = data.subject_entity['start_idx'], data.subject_entity['end_idx']
      obj_start_idx, obj_end_idx = data.object_entity['start_idx'], data.object_entity['end_idx']
      
      sub_type = data.subject_entity['type']
      obj_type = data.object_entity['type']
      
      s = data.sentence
      
      if sub_start_idx < obj_start_idx:
          res = [
              s[:sub_start_idx],
              f"[SUB:{sub_type}]" + s[sub_start_idx:sub_end_idx+1] + "[/SUB]",
              s[sub_end_idx+1:obj_start_idx],
              f"[OBJ:{obj_type}]" + s[obj_start_idx:obj_end_idx+1] + "[/OBJ]",
              s[obj_end_idx+1:]
          ]
      else:
          res = [
              s[:obj_start_idx],
              f"[OBJ:{obj_type}]" + s[obj_start_idx:obj_end_idx+1] + "[/OBJ]",
              s[obj_end_idx+1:sub_start_idx],
              f"[SUB:{sub_type}]" + s[sub_start_idx:sub_end_idx+1] + "[/SUB]",
              s[sub_end_idx+1:]
          ]
      
      return ''.join(res)    
  
  ```

- [ ] [Stratified K-Fold 추가: 참고 사항](https://github.com/boostcampaitech2/klue-level2-nlp-15/blob/main/train_with_pororo.ipynb)

---

- [x] ~~Multitask Classification을 구현하고 싶다. (Duo classifier)~~
- [x] ~~KoElectra as backbone model~~

- [x] Data Augmentation
  - [x] KoEDA - Random Switching, 동의어 바꾸기 -> CSV로 공유해주시면 좋을 것 같음다 ㅎㅎㅎ
  - [x] pororo를 이용해서 round trip translation을 하고 싶다 -> 영진
  
  
