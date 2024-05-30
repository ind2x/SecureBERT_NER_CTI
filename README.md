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
    + 사이트에서 크롤링한 데이터 속에 유니코드(사진 기호 등)이 들어있는 경우에는 제거해줘야 하고 이미 라벨링이 된 경우 제거 후 해당 유니코드가 차지하는 글자 수만큼 그 자리에 공백 추가
    + 라벨링 시 단어 중간에 라벨링을 한 경우 변환 시 잘못 태그가 되므로 제거하거나 다시 라벨링 해줘야 함

<br>

+ 라벨링 사진

![labeling](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/1c4b3775-27a1-44ae-83da-e878cc218628)

![labels](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/c71ea948-e53c-4fff-8867-e93418c86603)

<br>

+ 딥러닝 모델 (사전학습 bert모델)
  + SecureBERT-Plus (https://huggingface.co/ehsanaghaei/SecureBERT_Plus)
  + chatgpt ui 형태의 사이트 제작 후 입력 값에 대한 모델 결과 체크

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
