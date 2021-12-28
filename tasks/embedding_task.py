import tensorflow as tf
from tensorflow_addons.layers import crf

import json
import os
from trainer.train_base import TrainBase
from data_processor.classifier_data_generator import ClassifierDataGenerator
from data_processor.ner_data_generator import NERDataGenerator

# from official.nlp.keras_nlp.encoders.bert_encoder import BertEncoder
from official.nlp.modeling.networks import BertEncoder
from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs
from data_processor.tokenizer import tokenizer
from official.nlp.configs import encoders
import dataclasses
from official.modeling.hyperparams import base_config
from official.core import base_task
from official.core import config_definitions as cfg
from official.core import task_factory
from typing import List, Optional, Tuple

from model.sentence_embedding import SentenceEmbedding

@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base span labeler configuration."""
  encoder: encoders.EncoderConfig = encoders.EncoderConfig()
  head_dropout: float = 0.1
  head_initializer_range: float = 0.02


@dataclasses.dataclass
class embeddingConfig(cfg.TaskConfig):
  """The model config."""
  # At most one of `init_checkpoint` and `hub_module_url` can be specified.
  init_checkpoint: str = ''
  hub_module_url: str = ''
  model: ModelConfig = ModelConfig()

  # The real class names, the order of which should match real label id.
  # Note that a word may be tokenized into multiple word_pieces tokens, and
  # we asssume the real label id (non-negative) is assigned to the first token
  # of the word, and a negative label id is assigned to the remaining tokens.
  # The negative label id will not contribute to loss and metrics.
  class_names: Optional[List[str]] = None
  train_data: cfg.DataConfig = cfg.DataConfig()
  validation_data: cfg.DataConfig = cfg.DataConfig()

class EmbeddingTask(object):
    '''
    抽取句子向量任务
    '''
    def __init__(self, config):
        self.config = config

    def build_model(self, seq_len):
        '''
        构建模型
        '''
        # encoder_network = encoders.build_encoder(encoders.EncoderConfig(bert=encoders.BertEncoderConfig(vocab_size=21128,
        #                                         num_layers=1)))
        bert_config = bert_configs.BertConfig.from_json_file(self.config['bert_config_path'])
        cfg = bert_config
        bert_encoder = BertEncoder(
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
        model = SentenceEmbedding(bert_encoder, seq_len, self.config)
        ckpt = tf.train.Checkpoint(model=bert_encoder)
        init_checkpoint = self.config['bert_model_path']

        ckpt.restore(init_checkpoint).assert_existing_objects_matched()

        return model

    def build_inputs(self, text):
        '''
        构建输入
        '''
        tokenize = tokenizer(self.config)

        batch_token_ids, batch_segment_ids, batch_mask = [], [], []
        word_ids, segment_ids, word_mask, seq_len = tokenize.encode(text)
        batch_token_ids.append(word_ids)
        batch_segment_ids.append(segment_ids)
        batch_mask.append(word_mask)
        inputs = dict(
            input_word_ids=batch_token_ids,
            input_mask=batch_mask,
            input_type_ids=batch_segment_ids,
        )

        infer_input = {
            "input_word_ids": tf.convert_to_tensor(inputs['input_word_ids']),
            "input_mask": tf.convert_to_tensor(inputs['input_mask']),
            "input_type_ids": tf.convert_to_tensor(inputs['input_type_ids']),
        }
        return infer_input, seq_len


    def inference_one(self, text):
        '''
        推理一条数据
        '''
        infer_inputs, seq_len = self.build_inputs(text)
        model = self.build_model(seq_len)
        outputs = model(infer_inputs)
        return outputs

if __name__=='__main__':
    with open("../model_configs/sentence_embedding.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    embedding = EmbeddingTask(config)
    text = '你好'
    result = embedding.inference_one(text)
    print(result)
