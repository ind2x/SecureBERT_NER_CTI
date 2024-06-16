## SecureBERT-Plus와 BiLSTM+CRF로 NER
---

Fine-Tuning 한다는 것은 사전학습된 SecureBERT-Plus에 우리가 만든 데이터셋을 추가로 학습시키는 것이고, NER 부분은 SecureBERT-Plus로 임베딩하고 결과를 BiLSTM+CRF에 넣어줘서 파라미터를 업데이트해주면 됨.

이 코드는 Fine-Tuning은 못했고, SecureBERT-Plus로 임베딩만 해준 뒤 BiLSTM+CRF로 넘겨주는 코드임.

<br>

결과가 실제 데이터의 라벨들이 이상하게 달려있어서 더 진행하진 않았고, 추후에 따로 개인적으로 해보면 좋을 듯.

<br>

![image](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/5acd8304-3f73-4c59-84bc-1629c7d4f64d)
