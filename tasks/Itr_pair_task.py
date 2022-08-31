from official.nlp.bert import tokenization
import tensorflow as tf
from official.nlp.configs import bert
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader

from official.nlp.tasks.tagging import TaggingTask
from trainer.train_base import TrainBase
from official.nlp.modeling.models import BertClassifier
import os
import json
from data_processor.text_match_data_generator import TextMatchDataGenerator
from official.nlp.modeling.networks import BertEncoder
from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs
from models.sim_bert import SimBert



class ItrTask(TrainBase):
    '''
    基于bert的分类任务
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = 'loss'
        super(ItrTask, self).__init__(task_config)
        self.data_generator = TextMatchDataGenerator(task_config)


    def build_model(self):
        '''
        构建模型
        '''
        # encoder_network = encoders.build_encoder(encoders.EncoderConfig(
        #     bert=encoders.BertEncoderConfig(vocab_size=21128)))
        encoder_network = self.build_encoder()
        model = SimBert(network=encoder_network, config=self.config)

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
            return_all_encoder_outputs=True)
        # ckpt = tf.train.Checkpoint(model=bert_encoder)
        # init_checkpoint = self.config['bert_model_path']
        # ckpt.restore(init_checkpoint).assert_existing_objects_matched()
        # bert_encoder.load_weights(init_checkpoint)
        return bert_encoder

    def build_losses(self, labels, model_outputs, metrics, aux_losses=None) -> tf.Tensor:
        '''
        构建损失
        '''
        with tf.name_scope('TextMatchTask/losses'):
            if self.config['model_name'] == 'simbert':
                # 构建对比损失
                y = tf.reshape(labels, (-1,))
                similarity = model_outputs['logits']
                cond = (similarity < self.config["neg_threshold"])
                zeros = tf.zeros_like(similarity, dtype=tf.float32)
                ones = tf.ones_like(similarity, dtype=tf.float32)
                squre_similarity = tf.square(similarity)
                neg_similarity = tf.where(cond, squre_similarity, zeros)

                pos_loss = y * (tf.square(ones - similarity) / 4)
                neg_loss = (ones - y) * neg_similarity
                losses = pos_loss + neg_loss
                loss = tf.reduce_mean(losses)
                return loss

            metrics = dict([(metric.name, metric) for metric in metrics])
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                     tf.cast(model_outputs['predictions'], tf.float32),
                                                                     from_logits=True)

            loss = tf.reduce_mean(losses)

            return loss

    def build_inputs(self, inputs):
        '''
        构建输入
        '''
        train_input = {
            "input_word_ids_a": tf.convert_to_tensor(inputs['input_word_ids_a']),
            "input_mask_a": tf.convert_to_tensor(inputs['input_mask_a']),
            "input_type_ids_a": tf.convert_to_tensor(inputs['input_type_ids_a']),
            "input_word_ids_b": tf.convert_to_tensor(inputs['input_word_ids_b']),
            "input_mask_b": tf.convert_to_tensor(inputs['input_mask_b']),
            "input_type_ids_b": tf.convert_to_tensor(inputs['input_type_ids_b']),
            "labels": inputs['input_target_ids']
        }
        return train_input

    def train_step(self,
                   inputs,
                   model: tf.keras.Model,
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
            loss = self.build_losses(inputs["labels"], outputs, metrics, aux_losses=None)

        tvars = model.trainable_variables
        grads = tape.gradient(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
        optimizer.apply_gradients(list(zip(grads, tvars)))
        labels = inputs['labels']
        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs['predictions'])
            logs.update({m.name: m.result() for m in model.metrics})
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs['predictions'])
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def validation_step(self, inputs, model: tf.keras.Model, metrics=None):
        '''
        验证集验证模型
        :param input:
        :param model:
        :return:
        '''
        labels = inputs['labels']
        outputs = self.inference_step(inputs, model)
        loss = self.build_losses(labels, outputs, metrics, aux_losses=model.losses)

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs['predictions'])
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs['predictions'])
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def build_metrics(self, training=None):
        '''
        构建评价指标
        :param training:
        :return:
        '''
        # del training
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='text_match_metrics')
        ]

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
    with open("../model_configs/sim_bert.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    Itr_pair = ItrTask(config)

    model = Itr_pair.build_model()
    bert_encoder = Itr_pair.build_encoder()
    ckpt = tf.train.Checkpoint(model=bert_encoder)
    init_checkpoint = config['bert_model_path']
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    # config = models.get_config()
    # Itr_pair.train(model)
    print(model.layers)


