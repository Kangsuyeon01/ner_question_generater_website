from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf

label_dict = {'PER_B': 0, 'DAT_B': 1, '-': 2, 'ORG_B': 3, 'CVL_B': 4, 'NUM_B': 5, 'LOC_B': 6, 'EVT_B': 7, 'TRM_B': 8, 'TRM_I': 9, 'EVT_I': 10, 'PER_I': 11, 'CVL_I': 12, 'NUM_I': 13, 'TIM_B': 14, 'TIM_I': 15, 'ORG_I': 16, 'DAT_I': 17, 'ANM_B': 18, 'MAT_B': 19, 'MAT_I': 20, 'AFW_B': 21, 'FLD_B': 22, 'LOC_I': 23, 'AFW_I': 24, 'PLT_B': 25, 'FLD_I': 26, 'ANM_I': 27, 'PLT_I': 28, '[PAD]': 29}
index_to_tag = {v:k for k,v in label_dict.items()} # 키-값 쌍 변경, 인덱스(키)로 태그(값) 찾기

class F1score(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def sequences_to_tags(self, label_ids, pred_ids):
      label_list = []
      pred_list = []

      for i in range(0, len(label_ids)):
        label_tag = []
        pred_tag = []

        for label_index, pred_index in zip(label_ids[i], pred_ids[i]):
          if label_index != -100:
            label_tag.append(index_to_tag[label_index])
            pred_tag.append(index_to_tag[pred_index])
        
        label_list.append(label_tag)
        pred_list.append(pred_tag)

      return label_list, pred_list

    def on_epoch_end(self, epoch, logs={}):

      y_predicted = self.model.predict(self.X_test)
      y_predicted = np.argmax(y_predicted, axis = 2)

      label_list, pred_list = self.sequences_to_tags(self.y_test, y_predicted)

      score = f1_score(label_list, pred_list)
      print(' - f1: {:04.2f}'.format(score * 100))
      print(classification_report(label_list, pred_list))