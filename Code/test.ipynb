import random
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchcrf import CRF

# 데이터 읽기 함수
def read_conll_format(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                word, pos, tag = line.strip().split()
                sentence.append((word, pos, tag))
            else:
                sentences.append(sentence)
                sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


# 파일 경로 및 데이터 읽기
print("[*] Read File........................")
file_path = './all.conll'
sentences = read_conll_format(file_path)

# 데이터 셔플 및 분할
random.seed(42)
random.shuffle(sentences)

train_size = int(len(sentences) * 0.7)
val_size = int(len(sentences) * 0.15)
test_size = len(sentences) - train_size - val_size

train_sentences = sentences[:train_size]
val_sentences = sentences[train_size:train_size + val_size]
test_sentences = sentences[train_size + val_size:]

print(f"    [*] Train size: {len(train_sentences)}  .................")
print(f"    [*] Validation size: {len(val_sentences)}  ...............")
print(f"    [*] Test size: {len(test_sentences)}  ...................")

# 태그 설정
tags = ['B-API', 'I-API', 'B-Attack_Tool', 'I-Attack_Tool', 'B-Attacker', 'I-Attacker', 'B-CVE_Number', 'I-CVE_Number', 'B-DATE', 'I-DATE',
        'B-DomainName', 'I-DomainName', 'B-File', 'I-File', 'B-File_Paths', 'I-File_Paths', 'B-IP', 'I-IP', 'B-Location', 'I-Location',
        'B-MITRE_TAC', 'I-MITRE_TAC', 'B-MITRE_TECH', 'I-MITRE_TECH', 'B-PLATFORM', 'I-PLATFORM', 'B-PROTOCOL', 'I-PROTOCOL', 'B-SOFTWARE', 'I-SOFTWARE', 'B-URL', 'I-URL', 'O']
tag_to_idx = {tag: idx for idx, tag in enumerate(tags)}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

# 데이터셋 클래스 정의
tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT_Plus")

class NERDataset(Dataset):
    def __init__(self, sentences, tokenizer, tag_to_idx):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words = [word for word, _, _ in sentence]
        tags = [tag for _, _, tag in sentence]
        tokens = []
        label_ids = []

        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_ids.extend([self.tag_to_idx[tag]] * len(word_tokens))

        tokens = ['<s>'] + tokens + ['</s>']
        label_ids = [self.tag_to_idx['O']] + label_ids + [self.tag_to_idx['O']]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        return {
            'sentence': sentence,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_ids
        }


# 데이터셋 및 데이터로더 생성
print("[*] Make train, validation, test Dataset ........................")

train_dataset = NERDataset(train_sentences, tokenizer, tag_to_idx)
val_dataset = NERDataset(val_sentences, tokenizer, tag_to_idx)
test_dataset = NERDataset(test_sentences, tokenizer, tag_to_idx)

# Custom collate function for padding
def collate_fn(batch):
    input_ids = [torch.tensor(x['input_ids']) for x in batch]
    attention_mask = [torch.tensor(x['attention_mask']) for x in batch]
    labels = [torch.tensor(x['labels']) for x in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=tag_to_idx['O'])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

# 모델 정의
class NERModel(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_labels):
        super(NERModel, self).__init__()
        self.bert = bert_model  # SecureBERT 모델 초기화
        self.lstm = nn.LSTM(bert_model.config.hidden_size,
                            hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)  # BERT 임베딩
        sequence_output = outputs.last_hidden_state  # BERT의 마지막 레이어 출력
        lstm_out, _ = self.lstm(sequence_output)  # BiLSTM 레이어
        emissions = self.fc(lstm_out)  # Fully Connected 레이어
        attention_mask = attention_mask.bool()  # 수정된 부분: attention_mask를 bool 타입으로 변환
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask,
                             reduction='mean')  # CRF 레이어에서의 손실 계산
            return loss
        else:
            return self.crf.decode(emissions, mask=attention_mask)  # CRF 디코딩


# 모델 초기화 및 학습 설정
hidden_dim = 256
num_labels = len(tags)
bert_model = RobertaModel.from_pretrained("ehsanaghaei/SecureBERT_Plus")
model = NERModel(bert_model, hidden_dim, num_labels)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# 학습 함수 정의
def train(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

# 평가 함수 정의
def evaluate(model, dataloader):
    model.eval()
    predictions = []
    true_labels = []
    sentences = []  # 문장을 저장할 리스트 추가

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            preds = model(input_ids, attention_mask)
            predictions.extend(preds)
            true_labels.extend(labels)

    return predictions, true_labels  # 수정된 반환값


# 모델 저장 함수
def save_model(model, path):
    torch.save(model.state_dict(), path)

# 모델 로드 함수
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# 학습 과정
for epoch in range(1):
    print(f"[*] Epoch {epoch+1} ................")
    train(model, train_dataloader, optimizer)

    # 검증 데이터셋에 대해 평가
    print("\n[*] Evaluate on validation data ................")
    val_predictions, val_true_labels = evaluate(model, val_dataloader)

    # 검증 결과를 예시로 출력 (첫 번째 배치에 대해)
    for i in range(2):
        print("Word                 | True Label | Prediction")
        print("------------------------------------------------------")

        for word, prediction, true_label in zip(val_sentences[i], [idx_to_tag[p] for p in val_predictions[i]], [idx_to_tag[t] for t in val_true_labels[i].tolist()]):
            print(f"{word[0]:<20} | {true_label:<10} | {prediction}")
        print()
        
# 모델 저장
# save_model(model, "./ner_model.pth")

# 테스트 데이터셋에 대해 평가
print("\n[*] Evaluate on test data ................")
test_predictions, test_true_labels = evaluate(model, test_dataloader)

# 테스트 결과를 예시로 출력 (첫 번째 배치에 대해)
for i in range(2):
    print("Word                 | True Label | Prediction")
    print("------------------------------------------------------")

    for word, prediction, true_label in zip(test_sentences[i], [idx_to_tag[p] for p in test_predictions[i]], [idx_to_tag[t] for t in test_true_labels[i].tolist()]):
        print(f"{word[0]:<20} | {true_label:<10} | {prediction}")
    print()
