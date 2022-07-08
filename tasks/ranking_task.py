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
from data_processor.text_match_data_generator_v2 import TextMatchDataGeneratorV2
from official.nlp.modeling.networks import BertEncoder
from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs
from models.ranking import Ranking
import numpy as np



class RankingTask(TrainBase):
    '''
    基于bert的分类任务
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = 'loss'
        super(RankingTask, self).__init__(task_config)
        self.data_generator = TextMatchDataGeneratorV2(task_config)


    def build_model(self):
        '''
        构建模型
        '''
        # encoder_network = encoders.build_encoder(encoders.EncoderConfig(
        #     bert=encoders.BertEncoderConfig(vocab_size=21128)))
        encoder_network = self.build_encoder()
        model = Ranking(network=encoder_network, config=self.config)

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

    def lambda_rank_loss(self, scores, labels):
        '''
        lambda rank损失
        '''
        #delta_lambda计算
        rank = tf.range(1., tf.cast(self.config['num_samples'], dtype=tf.float32) + 1)
        rank = tf.tile(rank, [self.config['batch_size']])
        rank = tf.reshape(rank, tf.shape(labels))
        rel = 2 ** labels - 1
        sorted_label = tf.sort(labels, direction='DESCENDING')
        sorted_rel = 2 ** sorted_label - 1
        cg_discount = tf.math.log(1. + rank)
        dcg_m = rel / cg_discount
        dcg = tf.reduce_sum(dcg_m)
        stale_ij = dcg_m
        new_ij = rel / tf.transpose(cg_discount, perm=[0, 2, 1])
        stale_ji = tf.transpose(stale_ij, perm=[0, 2, 1])
        new_ji = tf.transpose(new_ij, perm=[0, 2, 1])
        #new dcg
        dcg_new = dcg - stale_ij + new_ij - stale_ji + new_ji
        #delta dcg
        dcg_max = tf.reduce_sum(sorted_rel / cg_discount)
        ndcg_delta = tf.abs(dcg_new - dcg) / dcg_max

        #
        s_i_minus_s_j = scores - tf.transpose(scores, perm=[0, 2, 1])
        #上三角矩阵
        mask1 = tf.linalg.band_part(ndcg_delta, 0, -1)
        #下三角矩阵
        mask2 = tf.linalg.band_part(s_i_minus_s_j, -1, 0)
        _loss = mask1 * tf.transpose(mask2, perm=[0, 2, 1])
        loss = tf.reduce_sum(_loss)
        return loss


    def build_losses(self, labels, model_outputs, metrics, aux_losses=None) -> tf.Tensor:
        '''
        构建NDCG损失
        '''
        def _ndcg(rank, relations):
            _dcg = [(np.power(2, relations[i]) - 1) / np.log2(rank[i] + 1) for i in range(len(relations))]
            _sort_similarity = sorted(relations, reverse=True)
            _idcg = [(np.power(2, _sort_similarity[i]) - 1) / np.log2(rank[i] + 1) for i in range(len(_sort_similarity))]
            _ndcg = tf.reduce_sum(_dcg) / tf.reduce_sum(_idcg)
            return _ndcg



        with tf.name_scope('TextMatchTask/lambdas'):
            # 构建ndcg损失
            tf.transpose(labels)
            y = tf.reshape(labels, [self.config['batch_size'], 1, self.config['num_samples']])
            similarity = model_outputs['logits']

            _relations = tf.keras.layers.Activation(tf.nn.sigmoid)(similarity)
            relations = tf.reshape(_relations[:, :, 1], tf.shape(y))
            # rank = [i for i in range(1, self.config['num_samples']+1)]
            # _dcg = [(np.power(2,relations[i])-1) / np.log2(rank[i]+1) for i in range(len(relations))]
            # _sort_similarity = [sorted(item, reverse=True) for item in _dcg]
            # _idcg = [(tf.pow(2,r)-1) / (tf.math.log(rank+1)/tf.math.log(2)) for r in _sort_similarity]
            # _ndcg = tf.reduce_sum(_dcg) / tf.reduce_sum(_idcg)
            # ndcg = [_ndcg(rank, relations[i]) for i in range(len(relations))]

            # y = [_ndcg(rank, y[i]) for i in range(len(y))]
            metrics = dict([(metric.name, metric) for metric in metrics])
            # losses = tf.keras.losses.sparse_categorical_crossentropy(y,
            #                                                          tf.cast(ndcg, tf.float32),
            #                                                          from_logits=True)

            loss = self.lambda_rank_loss(relations, y)

            return loss

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
            self.process_metrics(metrics, tf.reshape(labels, (-1,1)), tf.reshape(outputs['predictions'], (-1,1)))
            logs.update({m.name: m.result() for m in model.metrics})
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, tf.reshape(labels, (-1,1)), tf.reshape(outputs['predictions'], (-1,1)))
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
            self.process_metrics(metrics, tf.reshape(labels, (-1,1)), tf.reshape(outputs['predictions'], (-1,1)))
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, tf.reshape(labels, (-1,1)), tf.reshape(outputs['predictions'], (-1,1)))
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
    with open("../model_configs/ranking.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    Itr_pair = RankingTask(config)

    model = Itr_pair.build_model()
    bert_encoder = Itr_pair.build_encoder()
    ckpt = tf.train.Checkpoint(model=bert_encoder)
    init_checkpoint = config['bert_model_path']
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    # config = models.get_config()
    Itr_pair.train(model)
    # print(model.layers)


