# 2024-05-29 작성 완료
# Doccano로 Sequence Labeling 프로젝트에서 라벨링을 한 후 jsonl 형태로 데이터 추출
# 추출된 jsonl 데이터셋을 NER 학습 데이터셋 형태인 CoNLL 형태로 변환하는 코드
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# NLTK를 위해 필요한 패키지 다운로드
nltk.download('punkt')    # nltk tokenizer model 
nltk.download('averaged_perceptron_tagger') # nltk pos tag model 

def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def jsonl_to_conll_with_pos(jsonl_data):
    conll_lines = []

    for entry in jsonl_data:
        text = entry['text']
        labels = entry['label']

        # 문장을 \n 기준으로 분리합니다.
        # \n이 여러개가 있는 경우는 어떨지 확인해보았고, 문제 없이 제대로 태그를 해주는 것을 확인하였음
        sentences = text.split('\n')
        current_position = 0

        # 각 문장에 대해 처리합니다.
        for sentence in sentences:
            # 토큰화
            # 토큰은 스페이스를 기준으로 단어로 토큰화가 됨
            # 특수기호들도 따로 토큰화가 됨 ex) Russia's => Russia와 '로 토큰화 / @AnonNews -> @, AnonNews로 토큰화
            # 만약 라벨이 단어 중간에 되어 있는 경우 에러 발생됨 ex) All-Russia에서 Russia만 라벨링된 경우에 다시 전체를 라벨링 해줘야 함
            # 단어 전체 혹은 단어 첫 번째부터 되는 경우는 괜찮음 ex) Russian에서 Russia만 라벨링된 경우는 ok
            tokens = word_tokenize(sentence)
            # POS 태깅
            tagged_tokens = pos_tag(tokens)

            # 모든 토큰에 대해 O 태그 설정 
            bio_tags = ['O'] * len(tokens)

            # 토큰에 라벨 설정
            for start, end, entity in labels:
                if start >= current_position and end <= current_position + len(sentence):
                    # 라벨의 시작 인덱스와 끝 인덱스 찾기
                    relative_start = start - current_position
                    relative_end = end - current_position
                    # 기존 문장 속 라벨이 설정된 단어 찾음
                    entity_text = sentence[relative_start:relative_end]

                    # 토큰화된 문장에서 라벨의 위치 탐색
                    start_token_idx = len(word_tokenize(sentence[:relative_start]))
                    end_token_idx = start_token_idx + len(word_tokenize(entity_text))

                    # 토큰화된 문장에서 태그를 달아줘야 할 단어에 B 태그 설정
                    if start_token_idx < len(bio_tags):
                        bio_tags[start_token_idx] = 'B-' + entity

                    # I 태그 설정
                    for i in range(start_token_idx + 1, min(end_token_idx, len(bio_tags))):
                        bio_tags[i] = 'I-' + entity

            # CoNLL 형태로 저장
            for (word, pos), bio_tag in zip(tagged_tokens, bio_tags):
                conll_lines.append(f"{word} {pos} {bio_tag}")
            conll_lines.append("")  # 한 문장이 끝났으므로 구분을 위한 \n 추가

            # 문장이 끝났으므로 다음 문장의 인덱스 값으로 설정
            current_position += len(sentence) + 1

    return "\n".join(conll_lines)

# JSONL 파일 경로
jsonl_file_path = 'all.jsonl'

# JSONL 파일을 읽음
jsonl_data = read_jsonl_file(jsonl_file_path)

# JSONL 데이터를 CoNLL 형식으로 변환
conll_output_with_pos = jsonl_to_conll_with_pos(jsonl_data)

# 변환된 CoNLL 형식 데이터를 파일에 저장
conll_file_path = 'all.conll'
with open(conll_file_path, 'w', encoding='utf-8') as f:
    f.write(conll_output_with_pos)

print(f"CoNLL formatted data with POS tags has been saved to {conll_file_path}")
