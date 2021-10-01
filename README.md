[klue-level2-nlp-15](https://github.com/boostcampaitech2/klue-level2-nlp-15)

- [ ] R-BERT 처럼 entity_1, entity_2, [CLS]에서 hidden state vector을 뽑아내서 concat하기 <이열~> -> 영진, 현수
- [ ] Multitask Classification을 구현하고 싶다. (Duo classifier) -> 재영
- [ ] KoELECTRA에 맞게 Discriminator Model (RNN, fc 2개 등등 뒤에 붙이는 거)랑 Trainer까지!! 👍👍👍 (Dataset은 똑같을 것 같은데) -> 영진
  - [ ] token_len = 10, hidden_feature vectors = 10 x 768 -> [CLS] 사용?
- [x] AdamP 사용하고 싶다.
- [x] Bidirectional LSTM을 얹은 모델
- [ ] collate_fn 만들어서 batch마다 max length 정해서 125보다 더 작은 length인 경우에는 더 짧게 해서 학습시키려고 했습니다 -> max_length를 구해서 하면 버리는 데이터도 없고 좋을것같아요 아예 데이터셋에서 batch를 묶어줄 때, 비슷한 token 개수(length)인 것들을 모아서 뽑는 방식도 있더라고요! 이걸 uniform length batching이라고 하는 것 같습니다. 어제 성욱님이 token length distribution이 class 별로도 유사하다는 걸 보여주신 덕분에 uniform length batching도 걱정없이 쓸 수 있을 것 같습니다 -> 연주
  - [ ] 준홍님이 BERT 내부에서 별도로 처리하는 방식이 있다고 언급함. 4강에서 Dataset에서 짧은 문장에 대해서 학습할 수 있도록 짧은 문장도 넣는다.
- [ ] Train set을 입력하고 Randomly Mask 씌워서 Pretrain을 하고 싶다. 나만의 작은 KLUE-BERT 만들기 -> 준홍 🤗
  - [ ] Overfit이 일어날 확률이 좀 있어보이는데.
- [ ] Data Augmentation
  - [ ] KoEDA - Random Switching, 동의어 바꾸기 -> CSV로 공유해주시면 좋을 것 같음다 ㅎㅎㅎ
  - [ ] pororo를 이용해서 round trip translation을 하고 싶다 -> 영진
- [ ] K-Fold 돌리고 있는데, 궁금한 게 K-Fold를 돌리는 게 여러가지 방법이 있을 거라 생각하거든요. 모델을 GPU에 올렸다가 내렸다가 하는 방식으로 학습할 수 있을 것 같고. **Model을 .pt 파일로 저장했다가 거기에서 불러다가 쓰는 경우가 있을 것 같고.** 제가 하는 건 Prediction을 구한다 csv 단에서 ensemble하는 형식인 거거든요. trainer을 갖고 model을 저장하지 않고 싶은데, 잘 안 나오더라고요. 그래서 처음부터 trainer을 사용하지 않고 코드를 짰거든요. trainer 함수랑 똑같은 함수를 짰는데, 합칠 걸 생각하니까 어떻게 하는 게 좋으려나… -> Model 파일을 checkpoint 형식으로 저장해서 불러다가 쓰는 게 더 좋을 듯 하다.
- [ ] Trainer가 귀찮은 게, save pretrained를 사용하면 폴더 형태로 계속 저장이 되더라고요. .pth로 하면 파일형식으로 바로 되는데...
  - [ ] config 파일 있지 않나요. state_dict = True인가? 불러올 때 config랑 같이 불러와야 한다고 생각해서 좀 헷갈렸었거든요.
