import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from transformers import BertTokenizer
from transformers import *
import model
import evaluate
from keras.models import load_model


import warnings
warnings.filterwarnings('ignore')

from silence_tensorflow import silence_tensorflow

silence_tensorflow()


def file_open(filePath):
    f =open(filePath,"rb")
    data = pickle.load(f)
    f.close()
    return data



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokened_data = file_open('capstone2/bert_ner/data/token_data.pkl') 
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokened_data]
    labels = [token_label_pair[1] for token_label_pair in tokened_data]

    ## padding 
    # print(np.quantile(np.array([len(x) for x in tokenized_texts]), 0.975)) # 문장의 길이가 상위 2.5%(88) 인 지점
    max_len = 88
    bs = 32

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=max_len, dtype = "int", value=tokenizer.convert_tokens_to_ids("[PAD]"), truncating="post", padding="post")
    label_dict = {'PER_B': 0, 'DAT_B': 1, '-': 2, 'ORG_B': 3, 'CVL_B': 4, 'NUM_B': 5, 'LOC_B': 6, 'EVT_B': 7, 'TRM_B': 8, 'TRM_I': 9, 'EVT_I': 10, 'PER_I': 11, 'CVL_I': 12, 'NUM_I': 13, 'TIM_B': 14, 'TIM_I': 15, 'ORG_I': 16, 'DAT_I': 17, 'ANM_B': 18, 'MAT_B': 19, 'MAT_I': 20, 'AFW_B': 21, 'FLD_B': 22, 'LOC_I': 23, 'AFW_I': 24, 'PLT_B': 25, 'FLD_I': 26, 'ANM_I': 27, 'PLT_I': 28, '[PAD]': 29}
    tags = pad_sequences([lab for lab in labels], maxlen=max_len, value=label_dict["[PAD]"], padding='post',\
                     dtype='int', truncating='post')
    
    # Attention mask
    attention_masks = np.array([[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in input_ids])
    
    # 'token_type_ids' ( [CLS]와 [SEP]를 구분해줌 )
    token_type_ids = np.array([np.zeros(len(i)) for i in input_ids], dtype=int)
    
    # train 데이터에서 10% 만큼을 validation 데이터로 분리
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=222, test_size=0.1)

    # Atteion mask train-test data split
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=222, test_size=0.1)

    # 'token_type_ids' split
    tr_token_type, val_token_type,_,_ = train_test_split(token_type_ids, input_ids,
                                             random_state=222, test_size=0.1)

    X_train, y_train = (tr_inputs,tr_masks,tr_token_type), tr_tags
    X_test, y_test =  (val_inputs,val_masks,val_token_type), val_tags

    model = model.modeling(model_name='bert-base-multilingual-cased',
                                  tag_size=30)

    f1_score_report = evaluate.F1score(X_test, y_test)

    model.fit(
    X_train, y_train, epochs=20, batch_size=8,
    callbacks = [f1_score_report]
)   

    model.save_weights('capstone2/bert_ner/save_model/model_ADAM_8_20')


