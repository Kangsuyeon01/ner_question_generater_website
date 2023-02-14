from data_load import file_load, tag_split
from tokenization import tokenize_and_preserve_labels, tag_tokenize

import pickle

def file_save(f_list):
    filePath = 'capstone2/bert_ner/data/ko_bert_token_data.pkl'
    f = open(filePath,"wb")
    pickle.dump(f_list,f)
    f.close()
   

def file_open(filePath):
    f =open(filePath,"rb")
    data = pickle.load(f)
    f.close()
    return data



if __name__ == "__main__":
    file_path = "capstone2/bert_ner/data/train_data.txt"
    tagged_sentences = file_load(file_path)
    sentences, ner_tags = tag_split(tagged_sentences)
    idx_ner_tags,index_to_ner,tag_size = tag_tokenize(ner_tags)
    print(len(sentences))
    print(len(idx_ner_tags))
    
    # bert input process

    # tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, idx_ner_tags)]
    # print("end!")
    # file_save(tokenized_texts_and_labels)




