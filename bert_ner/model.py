import tensorflow as tf
from transformers import *

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

def compute_loss(labels, logits):

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
  active_loss = tf.reshape(labels, (-1,)) != -100
  reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
  labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)

  return loss_fn(labels, reduced_logits)

def modeling(model_name, tag_size):
    model = TFBertForTokenClassification(model_name, tag_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=compute_loss)
    return model
