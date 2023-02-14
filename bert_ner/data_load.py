import os
import re

# 파일 로드하고 텍스트파일 전처리하기
def file_load(file_path):
    tagged_sentences = []
    sentences = []
    with open(file_path,"r") as f:
        for line in f:
            if len(line) == 0 or line[0] == "\n":
                if len(sentences) > 0:
                    tagged_sentences.append(sentences)
                    sentences = []
            splits = line.rstrip('\n').split('\t')
            if len(splits) == 3:
                sentences.append([splits[1],splits[-1]])
    return tagged_sentences

# 전처리한 데이터를 바탕으로 단어와 태그 나누기
def tag_split(tagged_sentences):
    sentences, ner_tags = [],[]
    for tagged_sentence in tagged_sentences:
        sentence, tag_info = zip(*tagged_sentence)
        cng_sentence = []
        cng_sentence.append("[CLS]")
        tag_info = list(tag_info)
        tag_info.insert(0,'-')
        for str in sentence:
            # 한글, 영어, 소문자, 대문자 이외의 단어 제거
             str = re.sub(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]', "", str)
             cng_sentence.append(str)
             
        cng_sentence.append("[SEP]")
        tag_info.append('-')

        sentences.append(cng_sentence)
        ner_tags.append(tag_info)

    return sentences, ner_tags


if __name__ == "__main__":
    file_path = "data/train_data.txt"
    tagged_sentences = file_load(file_path)
    sentences, ner_tags = tag_split(tagged_sentences)

