from konlpy.tag import Mecab
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from seqeval.metrics import f1_score, classification_report
from transformers import shape_list, BertTokenizer, TFBertModel
from tensorflow import keras
import transformers
import os
import re

from silence_tensorflow import silence_tensorflow

silence_tensorflow()


mecab = Mecab()
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

label_dict = {'PER_B': 0, 'DAT_B': 1, '-': 2, 'ORG_B': 3, 'CVL_B': 4, 'NUM_B': 5, 'LOC_B': 6, 'EVT_B': 7, 'TRM_B': 8, 'TRM_I': 9, 'EVT_I': 10, 'PER_I': 11, 'CVL_I': 12, 'NUM_I': 13, 'TIM_B': 14, 'TIM_I': 15, 'ORG_I': 16, 'DAT_I': 17, 'ANM_B': 18, 'MAT_B': 19, 'MAT_I': 20, 'AFW_B': 21, 'FLD_B': 22, 'LOC_I': 23, 'AFW_I': 24, 'PLT_B': 25, 'FLD_I': 26, 'ANM_I': 27, 'PLT_I': 28, '[PAD]': 29}
index_to_tag = {v:k for k,v in label_dict.items()} # 키-값 쌍 변경, 인덱스(키)로 태그(값) 찾기



class TFBertForTokenClassification(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super(TFBertForTokenClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_output = outputs[0]
        prediction = self.classifier(all_output)

        return prediction


def modeling(model_name, tag_size):
    model = TFBertForTokenClassification(model_name, tag_size)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=compute_loss)
    return model

def compute_loss(labels, logits):

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  active_loss = tf.reshape(labels, (-1,)) != -100
  reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
  labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

  return loss_fn(labels, reduced_logits)

def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in tqdm(examples):
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            label_mask.extend([0]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            label_mask = label_mask[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]

        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask


        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(len(label_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_masks = np.asarray(label_masks, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks

    
def ner_prediction(examples, max_seq_len, tokenizer, lang='ko'):
  examples = [remove_stopwords(sentence) for sentence in examples] # 불용어 제거
  if lang == 'ko':
    examples = [mecab.morphs(sent) for sent in examples]
  else:
    examples = [sent.split() for sent in examples]

  X_pred, label_masks = convert_examples_to_features_for_prediction(examples, max_seq_len=88, tokenizer=tokenizer)
  model = model_load()
  y_predicted = model.predict(X_pred)
  y_predicted = np.argmax(y_predicted, axis = 2)

  pred_list = []
  result_list = []

  for i in range(0, len(label_masks)):
    pred_tag = []
    for label_index, pred_index in zip(label_masks[i], y_predicted[i]):
      if label_index != -100:
        pred_tag.append(index_to_tag[pred_index])

    pred_list.append(pred_tag)

  for example, pred in zip(examples, pred_list):
    one_sample_result = []
    for one_word, label_token in zip(example, pred):
      one_sample_result.append((one_word, label_token))
    result_list.append(one_sample_result)

  return result_list

def model_load():
    model = modeling(model_name='bert-base-multilingual-cased',
                                  tag_size=30)
    model.load_weights('/home/suyeon/code/capstone2/bert_ner/save_model/model_RMSprop_50')

    print("model_load성공!!")
    return model

def remove_stopwords(sentence):
  with open('/home/suyeon/code/capstone2/bert_ner/stop_words.txt',"r") as f:
    stop_words = [line.rstrip('\n') for line in f]
  result = []
  sentence = sentence.split(' ')
  for i in sentence:
    if i not in stop_words:
      i = re.sub(r'[^ㄱ-ㅣ가-힣0-9a-zA-Z.]', "", i) # 특수문자 제거
      result.append(i)
  return " ".join(result)

def pos_process(sentence):
  return mecab.pos(sentence)

if __name__ == "__main__":
    sent1 = '오리온스는 리그 최정상급 포인트가드 김동훈을 앞세우는 빠른 공수전환이 돋보이는 팀이다'
    sent2 = '이 다음에는 koBert를 fineturning할거야!'
    sent3 = '어찌 비로소 강수연이 큰 일을 다음에 얼만큼 내겠는가 이와 반대로 강수연은 너무나도 착할 뿐이다.'

    test_samples = [sent1, sent2, sent3]
    
    result_list = ner_prediction(test_samples, max_seq_len=88, tokenizer=tokenizer, lang='ko')
    # pos_list =list(pos_process(sentence) for sentence in test_samples)
    print(result_list)
    # print()
    # print(pos_list)
