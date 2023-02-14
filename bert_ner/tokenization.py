from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer



def tokenize_and_preserve_labels(sentence, text_labels):

  tokenizer = BertTokenizer.from_pretrained('skt/kobert-base-v1')
  tokenized_sentence = []
  labels = []

  for word, label in zip(sentence, text_labels):

    tokenized_word = tokenizer.tokenize(word)
    n_subwords = len(tokenized_word)

    tokenized_sentence.extend(tokenized_word)
    labels.extend([label] * n_subwords)

  return tokenized_sentence, labels

def tag_tokenize(ner_tags):
    tar_tokenizer = Tokenizer()
    tar_tokenizer.fit_on_texts(ner_tags)
    tag_size = len(tar_tokenizer.word_index) + 1
    tag_dict = tar_tokenizer.word_index
    tag_dict.update({"[PAD]":0})
    index_to_ner = {i:j for j, i in tag_dict.items()}

    # 영어 태그명을 인덱싱한 딕셔너리를 이용해 인덱스로 변환
    
    idx_ner_tags = []

    for tags in ner_tags:
        tagging_list = []
        for tag in tags:
            tag= tag.lower()
            tagging_list.append(tag_dict[tag])
        idx_ner_tags.append(tagging_list)

    return idx_ner_tags,index_to_ner,tag_size





