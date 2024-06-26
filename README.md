# SecureBERT_NER_CTI
---

2024 4-1학기 캡스톤

주제: 보안뉴스/취약점 보고서 등에서 위협 특징 정보(CTI)들을 자동추출하는 기술 개발

+ 크롤링 사이트
  + HackRead => 내 담당
    + Anonymous, CyberAttacks, CyberCrime, Malware 카테고리에서 수집, 총 250개
  + Hacker News
  + NIST
  + CISA

<br>

+ 라벨링
  + doccano 사용
  + 18개의 특징 정보들을 선정
  + doccano to CoNLL format 코드 작성
    + 라벨링 후 추출한 데이터셋 (jsonl)을 CoNLL 형태로 변환 
    + 라벨링 시 단어 중간에 라벨링을 한 경우 변환 시 잘못 태그가 되므로 제거하거나 다시 라벨링 해줘야 함
    + 사이트에서 크롤링한 데이터 속에 유니코드(사진 기호 등)이 들어있는 경우에는 제거해야 함
      + ex) 🇷🇺, 🔴, ⚠️, 📉
      + 이미 라벨링이 된 경우 제거 후 해당 유니코드가 차지하는 글자 수만큼 그 자리에 공백 추가
        + 글자 수만큼 공백으로 채워줘도 라벨 위치가 맞지 않는 경우도 있어서 수정 후 확인해줘야 함 

<br>

+ 딥러닝 모델 (사전학습 bert모델)
  + SecureBERT-Plus (https://huggingface.co/ehsanaghaei/SecureBERT_Plus)
  + SecureBERT-Plus와 NER은 BiLSTM+CRF 층 이용
    + SecureBERT-Plus 모델에 학습시켜야하는데 학습코드 인원들이 못해내서 그냥 BERT에 학습시킴
    + 이런 경우에는 SecureBERT-Plus 모델을 fine-tuning 시키지 않고 임베딩 용도로만 활용한 후 BiLSTM+CRF로 파라미터만 업데이트해줬어야 함
      + 코드는 짰지만 train, evaluate, test 데이터를 구분해줬는데 결과를 보니까 예측은 당연하고 내가 넣어준 데이터의 라벨이 이상하게 되어있음
      + train, evaluation, test 데이터로 분리해줄 때 이상하게 된건지.. 출력했을 때는 이상이 없는데 돌리고나서 확인해보면 이상하게 나옴

<br>

+ 라벨링 사진 (예시)

![labeling](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/1c4b3775-27a1-44ae-83da-e878cc218628)

![labels](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/c71ea948-e53c-4fff-8867-e93418c86603)

<br><hr style="border: 2px"><br>

## 피드백
---

+ 데이터 수집 후 라벨링 전 정제 과정이 필요
  + 문장이 끝날 때 '\n'으로 바로 끝나지 않고 공백+'\n'으로 끝나거나, '\n\n\n'와 같이 끝나는 경우
  + 뉴스를 수집하다보면 아이콘들이 같이 수집되버릴 수 있는데, 라벨링 하기 전에 제거해줘야 CoNLL 형태로 변환 시 문제 없이 진행됨

+ 라벨링 시 세부적인 가이드라인이 절실히 필요
  + 라벨을 정했으면 데이터 값의 형태들을 예상해보고 어떤 형태의 값들만 라벨링을 할 지 정해놔야 함.
  + 예를 들어, 날짜의 경우 년/월/일 형태와 년/월, 월/일, 년, 일 등의 형태가 있는데 어떤 형태들을 라벨링 해주고 안해줄지를 정해놔야 함. 

+ 토큰화 시 고려해야 되는 점들이 많은데 라벨링 전에 그러한 사항들을 고려해본 후 시작하는 것이 추후 CoNLL 형태로 변환할 때 도움이 될 듯

+ 이전에도 느꼈지만 리더쉽을 더 키워야 함
  + 이번에는 AI쪽으로 잘 알지 못할뿐더러 이 주제에 대해 경험해본 팀원이 있어서 조장을 하지 않았음
    + 상황 진전이 없거나 추진력이 필요할 때만 발휘되는 점이 아쉬움. 처음부터 맡아서 주도하면 더 좋게 흘러갈 수 있으니 다음에는 용기내어 시도해볼 것 
  + 잘 알지 못하는 내용들에 대해서 확신이 안들어 요구사항이나 제안을 잘 하지 못하였는데, 이러한 부분에서 틀리거나 잘못될 것 같더라도 의견을 내보는 것이 낫겠다고 느낌
  + 고로, 어차피 다들 잘 모르는 내용들이니 눈치보지 말고 의견을 제시하고 여러명의 의견을 합치는 것이 더 좋은 방식이라고 느낌

+ 발표 피드백 (교수님, 멘토님들 피드백 내용)
  + **발표 자신감이 필요**하다는 말을 또 듣게 됨.. 인턴할 때도 들었던 내용이라 아쉬움
    + 두 상황 모두 결과에 대한 확신 혹은 결과가 잘 나오지 않아 자신감이 없어서 그런지 목소리가 작았음
    + 인턴 때는 추가로 긴장으로 인해 떨려하는 부분이 많이 있었는데 이번 최종발표때는 그런 부분은 많이 완화돼어서 다행인 부분도 있었음

  + 현재 계획이 안되었을 때 "다른 방법으로 시도해보았고 그 결과는 이렇게 나왔다" 라는 내용이 있으면 더 좋겠다
    + 학습코드 담당은 아니지만 따로 작성을 해보긴 했는데 반영하진 않았음
    
