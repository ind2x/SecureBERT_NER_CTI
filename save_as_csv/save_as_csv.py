import csv

def split_file_by_commas(filename):
	with open(filename, 'r', encoding='utf-8') as file:
		lines = file.readlines()

	blocks = []
	current_block = []

	for line in lines:
		if line.strip() == ',,,':  # ,,,만 있는 줄을 기준으로 블록 나누기
			if current_block:
				blocks.append(current_block)
				current_block = []
		else:
			current_block.append(line.strip())

	if current_block:
		blocks.append(current_block)  # 마지막 블록 추가

	return blocks

def modify_first_element(blocks):
	for i, block in enumerate(blocks):
		if block:  # 블록이 비어있지 않은 경우
			first_element = block[0].split(',')
			first_element[0] = f"Sentence {i + 1}"
			block[0] = ','.join(first_element)

# 파일 경로 지정
input_filename = 'input.csv'  # 처음에 xlsx를 csv로 변환시킨 파일
output_filename = 'output.csv'  # 단순 파일 입출력으로 저장시킬 파일
output2_filename = 'res.csv'  # csv 파일 입출력으로 csv로 저장시킬 파일

# 파일을 읽고 분리된 블록을 가져오기
blocks = split_file_by_commas(input_filename)
# blocks 리스트에 대해 수정 수행
modify_first_element(blocks)
#print(blocks[960])

with open(output_filename, 'w', newline='', encoding='utf-8') as f:
	for i in range(len(blocks)):
		for block in blocks[i]:
			f.write(block + '\n')

# 기존 파일을 읽어와서 동일한 형식으로 다시 저장
with open(output_filename, 'r', newline='', encoding='utf-8-sig') as infile, \
  open(output2_filename, 'w', newline='', encoding='utf-8-sig') as outfile:
	reader = csv.reader(infile, quotechar='"', quoting=csv.QUOTE_ALL)
	writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
	for row in reader:
		writer.writerow(row)

print("end")
