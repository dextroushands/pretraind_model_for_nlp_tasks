# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Examples of SavedModel export for tf-serving."""

from absl import app
from absl import flags
import tensorflow as tf

from official.nlp.bert import bert_models
from official.nlp.bert import configs
from tasks.embedding_task import EmbeddingTask
from official.nlp.modeling.networks import BertEncoder
from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs
from data_processor.tokenizer import tokenizer
from models.sentence_embedding import SentenceEmbedding
import json


root_path = '/Users/donruo/Desktop/project/bert_tasks/'

flags.DEFINE_string("bert_config_file", root_path+'chinese_wwm_ext_L-12_H-768_A-12/v2/bert_config.json',
                    "Bert configuration file to define core bert layers.")
flags.DEFINE_string("model_checkpoint_path", root_path+'chinese_wwm_ext_L-12_H-768_A-12/v2/bert_model.ckpt-1',
                    "File path to TF model checkpoint.")
flags.DEFINE_string("export_path", root_path+'chinese_wwm_ext_L-12_H-768_A-12/serve/versions/1',
                    "Destination folder to export the serving SavedModel.")
flags.DEFINE_string("config_path", root_path+'model_configs/sentence_embedding.json', "embedding model configurations")

FLAGS = flags.FLAGS


class BertServing(tf.keras.Model):
  """Bert transformer encoder model for serving."""

  def __init__(self, config, bert_config, name_to_features, name="serving_model"):
    super(BertServing, self).__init__(name=name)

    cfg = bert_config
    self.bert_encoder = BertEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        activation=tf_utils.get_activation(cfg.hidden_act),
        dropout_rate=cfg.hidden_dropout_prob,
        attention_dropout_rate=cfg.attention_probs_dropout_prob,
        max_sequence_length=cfg.max_position_embeddings,
        type_vocab_size=cfg.type_vocab_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=cfg.initializer_range),
        embedding_width=cfg.embedding_size,
        return_all_encoder_outputs=True)
    self.model = SentenceEmbedding(self.bert_encoder, config)
    # ckpt = tf.train.Checkpoint(model=self.bert_encoder)
    # init_checkpoint = self.config['bert_model_path']
    #
    # ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    self.name_to_features = name_to_features

  def call(self, inputs):
    input_word_ids = inputs["input_word_ids"]
    input_mask = inputs["input_mask"]
    input_type_ids = inputs["input_type_ids"]
    infer_input = {
        "input_word_ids": input_word_ids,
        "input_mask": input_mask,
        "input_type_ids": input_type_ids,
    }
    encoder_outputs = self.model(
        infer_input)
    return encoder_outputs

  def serve_body(self, input_ids, input_mask=None, segment_ids=None):
    if segment_ids is None:
      # Requires CLS token is the first token of inputs.
      segment_ids = tf.zeros_like(input_ids)
    if input_mask is None:
      # The mask has 1 for real tokens and 0 for padding tokens.
      input_mask = tf.where(
          tf.equal(input_ids, 0), tf.zeros_like(input_ids),
          tf.ones_like(input_ids))

    inputs = dict(
        input_word_ids=input_ids, input_mask=input_mask, input_type_ids=segment_ids)
    return self.call(inputs)

  @tf.function
  def serve(self, input_ids, input_mask=None, segment_ids=None):
    outputs = self.serve_body(input_ids, input_mask, segment_ids)
    # Returns a dictionary to control SignatureDef output signature.
    return {"outputs": outputs}

  @tf.function
  def serve_examples(self, inputs):
    features = tf.io.parse_example(inputs, self.name_to_features)
    for key in list(features.keys()):
      t = features[key]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      features[key] = t
    return self.serve(
        features["input_word_ids"],
        input_mask=features["input_mask"] if "input_mask" in features else None,
        segment_ids=features["input_type_ids"]
        if "input_type_ids" in features else None)

  @classmethod
  def export(cls, model, export_dir):
    if not isinstance(model, cls):
      raise ValueError("Invalid model instance: %s, it should be a %s" %
                       (model, cls))

    signatures = {
        "serving_default":
            model.serve.get_concrete_function(
                input_ids=tf.TensorSpec(
                    shape=[None, None], dtype=tf.float32, name="inputs")),
    }
    if model.name_to_features:
      signatures[
          "serving_examples"] = model.serve_examples.get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name="examples"))
    tf.saved_model.save(model.model, export_dir=export_dir, signatures=signatures)


def main(_):
  config_path = FLAGS.config_path
  with open(config_path, 'r') as fr:
      config = json.load(fr)
  sequence_length = config['seq_len']
  if sequence_length is not None and sequence_length > 0:
    name_to_features = {
        "input_word_ids": tf.io.FixedLenFeature([sequence_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([sequence_length], tf.int64),
        "input_type_ids": tf.io.FixedLenFeature([sequence_length], tf.int64),
    }
  else:
    name_to_features = None
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  serving_model = BertServing(
      config=config, bert_config=bert_config, name_to_features=name_to_features)
  checkpoint = tf.train.Checkpoint(model=serving_model.bert_encoder)
  checkpoint.restore(FLAGS.model_checkpoint_path
                    ).assert_existing_objects_matched()
  '''.run_restore_ops()'''
  BertServing.export(serving_model, FLAGS.export_path)

def get_serving_predict(self, sentence):
    '''
    使用tf-serving加载模型
    :param sentence:
    :return:
    '''
    # docker
    # run - t - -rm - p 8500: 8500 \
    # - v "/Users/donruo/Desktop/project/search_algorithm/ranking/tf_ranking/examples/output/export/latest_exporter/1614153823/" \
    # - e MODEL_NAME = saved_model \
    # tensorflow / serving: 1.15.0 &
    sentence = list(sentence)
    sentence_ids = self.sentence_to_idx(sentence)
    # print(sentence_ids)
    embedded_words = []
    [embedded_words.append(self.word_vectors[i].tolist()) for i in sentence_ids]
    # print(len(embedded_words))
    # tf.contrib.util.make_tensor_proto(padding_sentence,
    #                                   dtype=tf.int64,
    #                                   shape=[1, 50]).SerializeToString())

    data = json.dumps({"signature_name": "classifier", "instances": [{"inputs": sentence_ids, "keep_prob": 1.0}]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/savedModel:predict',
                                  data=data, headers=headers)
    prediction = json.loads(json_response.text)
    print(prediction)

    return prediction

if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("model_checkpoint_path")
  flags.mark_flag_as_required("export_path")
  flags.mark_flag_as_required('config_path')
  app.run(main)
