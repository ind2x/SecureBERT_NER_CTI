## CoNLL 데이터셋을 txt가 아닌 csv로 저장
---

![image](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/e3ca6dd3-86f4-4265-8ec6-df303ab53ba9)

<br>

기존에 위에처럼 저장된 파일을 csv로 저장시키는 코드

csv는 텍스트에디터로 열었을 때 데이터를 볼 수 있기 때문에 csv 모듈을 사용하지 않아도 됨.

저장할 때만 메모장에 데이터를 옮긴 후 UTF-8(BOM) 인코딩 형태로만 저장시켜주면 특수기호나 표기가 다른 쌍따옴표 등도 잘 저장됨.
+ csv 모듈로 utf-8 인코딩을 적용해서 입출력을 해줘도 소용없었음

<br>

완전 자동화는 아님.

1. conll 내용을 빈 엑셀에 입력 후 [데이터 탭 - 텍스트 나누기 - 구분 기호로 분리됨, 공백, 일반] 선택 후 csv utf-8로 저장
2. 1번으로 input.csv 생성되면 코드 실행 후 res.csv 내용을 메모장으로 옮겨서 UTF-8(BOM) 인코딩 형태로 저장  

<br>

+ input.csv
  
![image](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/c579a348-ad7e-43f6-b964-6673beec0df7)

<br>

+ res.csv

![image](https://github.com/ind2x/SecureBERT_NER_CTI/assets/52172169/349a81e8-240d-4562-979e-7e81c2ac45bf)
