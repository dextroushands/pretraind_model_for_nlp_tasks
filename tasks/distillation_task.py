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
from models.knowledge_distiilation import Distill_model
from models.sim_bert import SimBert



class DistillTask(TrainBase):
    '''
    基于bert的知识蒸馏任务
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = 'loss'
        super(DistillTask, self).__init__(task_config)
        self.data_generator = TextMatchDataGenerator(task_config)


    def build_model(self):
        '''
        构建模型
        '''
        # encoder_network = encoders.build_encoder(encoders.EncoderConfig(
        #     bert=encoders.BertEncoderConfig(vocab_size=21128)))
        bert_config = bert_configs.BertConfig.from_json_file(self.config['bert_config_path'])
        encoder_network = self.build_encoder()
        teacher_network = SimBert(network=encoder_network, config=self.config)

        model = Distill_model(teacher_network=teacher_network, config=self.config, vocab_size=bert_config.vocab_size, word_vectors=None)

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
            if self.config['model_name'] == 'distill_model':
                # mse损失计算
                y = tf.reshape(labels, (-1,))
                student_soft_label = model_outputs['student_soft_label']
                teacher_soft_label = model_outputs['teacher_soft_label']
                mse_loss = tf.keras.losses.mean_squared_error(teacher_soft_label, student_soft_label)

                #ce损失计算
                similarity = model_outputs['student_hard_label']
                cond = (similarity < self.config["neg_threshold"])
                zeros = tf.zeros_like(similarity, dtype=tf.float32)
                ones = tf.ones_like(similarity, dtype=tf.float32)
                squre_similarity = tf.square(similarity)
                neg_similarity = tf.where(cond, squre_similarity, zeros)

                pos_loss = y * (tf.square(ones - similarity) / 4)
                neg_loss = (ones - y) * neg_similarity
                ce_loss = pos_loss+neg_loss
                losses = self.config['alpha']*mse_loss + (1-self.config['alpha'])*ce_loss
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
            "input_x_ids": tf.convert_to_tensor(inputs['input_word_ids_a']),
            "input_y_ids": tf.convert_to_tensor(inputs['input_word_ids_b']),
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
    with open("../model_configs/distill_bert.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    distill_pair = DistillTask(config)

    model = distill_pair.build_model()
    bert_encoder = distill_pair.build_encoder()
    ckpt = tf.train.Checkpoint(model=bert_encoder)
    init_checkpoint = config['bert_model_path']
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    # config = models.get_config()
    # new_model = tf.keras.Model(inputs=model.inputs[0:2], outputs=model.output['predictions'])
    # for layer in model.layers:
    #     if layer.name!='sim_bert':
    #         new_model.add(layer)
    distill_pair.train(model)
    # print(new_model.summary())


