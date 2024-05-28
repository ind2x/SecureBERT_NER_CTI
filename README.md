# SecureBERT_NER_CTI
---

2024 4-1학기 캡스톤

주제: 보안뉴스/취약점 보고서 등에서 위협 특징 정보(CTI)들을 자동추출하는 기술 개발

+ 크롤링 사이트
  + HackRead
  + Hacker News
  + NIST
  + CISA
+ 라벨링
  + doccano 사용
  + doccano to CoNLL format 코드 작성
+ 딥러닝 모델 (사전학습 bert모델)
  + SecureBERT-Plus (https://huggingface.co/ehsanaghaei/SecureBERT_Plus)
  + chatgpt ui 형태의 사이트 제작 후 입력 값에 대한 모델 결과 체크
