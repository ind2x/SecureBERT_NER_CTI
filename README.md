# SecureBERT_NER_CTI
---

2024 4-1학기 캡스톤

주제: 보안뉴스/취약점 보고서 등에서 위협 특징 정보(CTI)들을 자동추출하는 기술 개발

+ 크롤링 사이트
  + HackRead => 내 담당
  + Hacker News
  + NIST
  + CISA

<br>

+ 라벨링
  + doccano 사용
  + doccano to CoNLL format 코드 작성
    + 라벨링 후 추출한 데이터셋 (jsonl)을 CoNLL 형태로 변환 
    + 사이트에서 크롤링한 데이터 속에 유니코드(사진 기호 등)이 들어있는 경우에는 제거해줘야 하고 이미 라벨링이 된 경우 제거 후 해당 유니코드가 차지하는 글자 수만큼 그 자리에 공백 추가
    + 라벨링 시 단어 중간에 라벨링을 한 경우 변환 시 잘못 태그가 되므로 제거하거나 다시 라벨링 해줘야 함
<br>

+ 딥러닝 모델 (사전학습 bert모델)
  + SecureBERT-Plus (https://huggingface.co/ehsanaghaei/SecureBERT_Plus)
  + chatgpt ui 형태의 사이트 제작 후 입력 값에 대한 모델 결과 체크
