from official.nlp.bert import tokenization
import tensorflow as tf
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader

from official.nlp.tasks.tagging import TaggingTask
from trainer.train_base import TrainBase
from official.nlp.modeling.models import BertTokenClassifier
import os
import json
from data_processor.ner_data_generator import NERDataGenerator
from official.nlp.modeling.networks import BertEncoder
from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs

def _masked_labels_and_weights(y_true):
  """Masks negative values from token level labels.

  Args:
    y_true: Token labels, typically shape (batch_size, seq_len), where tokens
      with negative labels should be ignored during loss/accuracy calculation.

  Returns:
    (masked_y_true, masked_weights) where `masked_y_true` is the input
    with each negative label replaced with zero and `masked_weights` is 0.0
    where negative labels were replaced and 1.0 for original labels.
  """
  # Ignore the classes of tokens with negative values.
  mask = tf.greater_equal(y_true, 0)
  # Replace negative labels, which are out of bounds for some loss functions,
  # with zero.
  masked_y_true = tf.where(mask, y_true, 0)
  return masked_y_true, tf.cast(mask, tf.float32)

class NERTask(TrainBase):
    '''
    基于bert的分类任务
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = 'loss'
        super(NERTask, self).__init__(task_config)
        self.data_generator = NERDataGenerator(task_config)


    def build_model(self):
        '''
        构建模型
        '''
        # encoder_network = encoders.build_encoder(encoders.EncoderConfig(
        #     bert=encoders.BertEncoderConfig(vocab_size=21128)))
        encoder_network = self.build_encoder()



        model = BertTokenClassifier(network=encoder_network,
                                    num_classes=self.config['tag_categories'],
                                    dropout_rate=self.config['dropout_rate'],
                                    output='logits')
        # ckpt = tf.train.Checkpoint(models=models)

        # init_checkpoint = self.config['bert_model_path']

        # ckpt.restore(init_checkpoint).assert_existing_objects_matched()

        # models.load_weights(init_checkpoint).assert_existing_objects_matched()
        return model

    def build_encoder(self):
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
            return_all_encoder_outputs=False)
        # ckpt = tf.train.Checkpoint(model=bert_encoder)
        # init_checkpoint = self.config['bert_model_path']
        # ckpt.restore(init_checkpoint).assert_existing_objects_matched()
        # bert_encoder.load_weights(init_checkpoint)
        return bert_encoder

    def build_losses(self, labels, model_outputs, metrics, aux_losses=None) -> tf.Tensor:
        '''
        构建损失
        '''
        masked_labels, masked_weights = _masked_labels_and_weights(labels)
        metrics = dict([(metric.name, metric) for metric in metrics])
        losses = tf.keras.losses.sparse_categorical_crossentropy(masked_labels,
                                                                 tf.cast(model_outputs, tf.float32),
                                                                 from_logits=True)
        # metrics['losses'].update_state(losses)
        loss = losses
        numerator_loss = tf.reduce_sum(loss * masked_weights)
        denominator_loss = tf.reduce_sum(masked_weights)
        loss = tf.math.divide_no_nan(numerator_loss, denominator_loss)

        return loss

    def train_step(self,
                   inputs,
                   model:tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   metrics=None):
        '''
        进行训练，前向和后向计算
        :param inputs:
        :param model:
        :param optimizer:
        :param metrics:
        :return:
        '''
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            outputs = outputs[:, 1:self.config['seq_len'] + 1, :]
            loss = self.build_losses(labels=inputs['labels'], model_outputs=outputs, metrics=metrics, aux_losses=None)
        tvars = model.trainable_variables
        grads = tape.gradient(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
        optimizer.apply_gradients(list(zip(grads, tvars)))
        labels = inputs['labels']
        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
            logs.update({m.name: m.result() for m in model.metrics})
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def validation_step(self, inputs, model:tf.keras.Model, metrics=None):
        '''
        验证集验证模型
        :param input:
        :param model:
        :return:
        '''
        labels = inputs['labels']
        outputs = self.inference_step(inputs, model)
        outputs = outputs[:, 1:self.config['seq_len'] + 1, :]
        loss = self.build_losses(labels, outputs, metrics, aux_losses=model.losses)

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def build_inputs(self, inputs):
        '''
        构建输入
        '''
        train_input = {
            "input_word_ids": tf.convert_to_tensor(inputs['input_word_ids']),
            "input_mask": tf.convert_to_tensor(inputs['input_mask']),
            "input_type_ids": tf.convert_to_tensor(inputs['input_type_ids']),
            "labels": inputs['input_target_ids']
        }
        return train_input

    def build_metrics(self, training=None):
        '''
        构建评价指标
        :param training:
        :return:
        '''
        # del training
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='ner_metrics')
        ]

        # metrics = dict([(metric.name, metric) for metric in metrics])

        return metrics

    def check_exist_model(self, model):
        '''
        检查是否存在模型文件
        :return:
        '''
        # ckpt = tf.train.Checkpoint(models=models)
        init_checkpoint = os.path.join(self.config['ckpt_model_path'], self.config['model_name'])

        # ckpt.restore(init_checkpoint).assert_existing_objects_matched()
        model.load_weights(init_checkpoint).assert_existing_objects_matched()


if __name__=='__main__':
    with open("../model_configs/bert_ner.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    ner = NERTask(config)

    model = ner.build_model()
    bert_encoder = ner.build_encoder()
    ckpt = tf.train.Checkpoint(model=bert_encoder)
    init_checkpoint = config['bert_model_path']
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    # config = models.get_config()
    ner.train(model)


