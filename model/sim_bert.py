from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf
from official.nlp.modeling import layers
from official.nlp.modeling import networks


class SimBert(tf.keras.Model):
  """
  bert句子相似度模型
  """

  def __init__(self,
               network,
               config,
               initializer='glorot_uniform',
               dropout_rate=0.1,
               ):
      self._self_setattr_tracking = False
      self._network = network
      self._config = {
          'network': network,
          'initializer': initializer,
      }
      self.config = config
      #定义两个句子的输入
      # 定义输入
      word_ids_a = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids_a')
      mask_a = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask_a')
      type_ids_a = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids_a')
      word_ids_b = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids_b')
      mask_b = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask_b')
      type_ids_b = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids_b')
      input_a = [word_ids_a, mask_a, type_ids_a]
      input_b = [word_ids_b, mask_b, type_ids_b]

      #计算encoder
      outputs_a = network.predict_step(input_a)
      outputs_b = network.predict_step(input_b)

      cls_output_a = outputs_a[1]
      query_embedding_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output_a)

      cls_output_b = outputs_b[1]
      sim_query_embedding_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output_b)

      # 余弦函数计算相似度
      # cos_similarity余弦相似度[batch_size, similarity]
      query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_embedding_output), axis=-1), name='query_norm')
      sim_query_norm = tf.sqrt(tf.reduce_sum(tf.square(sim_query_embedding_output), axis=-1), name='sim_query_norm')

      dot = tf.reduce_sum(tf.multiply(query_embedding_output, sim_query_embedding_output), axis=-1)
      cos_similarity = tf.divide(dot, (query_norm * sim_query_norm), name='cos_similarity')
      self.similarity = cos_similarity

      # 预测为正例的概率
      cond = (self.similarity > self.config["neg_threshold"])
      pos = tf.where(cond, tf.square(self.similarity), 1 - tf.square(self.similarity))
      neg = tf.where(cond, 1 - tf.square(self.similarity), tf.square(self.similarity))
      predictions = [[neg[i], pos[i]] for i in range(self.config['batch_size'])]

      self.logits = self.similarity
      outputs = dict(logits=self.logits, predictions=predictions)

      super(SimBert, self).__init__(inputs=[input_a, input_b], outputs=outputs)

  @property
  def checkpoint_items(self):
      return dict(encoder=self._network)

  def get_config(self):
      return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
      return cls(**config)