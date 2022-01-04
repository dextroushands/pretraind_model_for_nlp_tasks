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
from data_processor.classifier_data_generator import ClassifierDataGenerator
from official.nlp.modeling.networks import BertEncoder
from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs



class ClassifierTask(TrainBase):
    '''
    基于bert的分类任务
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = 'loss'
        super(ClassifierTask, self).__init__(task_config)
        self.data_generator = ClassifierDataGenerator(task_config)


    def build_model(self):
        '''
        构建模型
        '''
        # encoder_network = encoders.build_encoder(encoders.EncoderConfig(
        #     bert=encoders.BertEncoderConfig(vocab_size=21128)))
        encoder_network = self.build_encoder()



        model = BertClassifier(network=encoder_network,
                               num_classes=self.config['num_classes'])
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
        if self.config['num_classes'] > 1:
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                     tf.cast(model_outputs, tf.float32),
                                                                     from_logits=True)
        else:
            losses = tf.keras.losses.categorical_crossentropy(labels,
                                                              tf.cast(model_outputs, tf.float32),
                                                              from_logits=True
                                                              )
        # metrics['losses'].update_state(losses)
        loss = tf.reduce_mean(losses)

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

    def build_metrics(self, training=None):
        '''
        构建评价指标
        :param training:
        :return:
        '''
        # del training
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='classifier_metrics')
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
    with open("../model_configs/classifier.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    classifier = ClassifierTask(config)

    model = classifier.build_model()
    bert_encoder = classifier.build_encoder()
    ckpt = tf.train.Checkpoint(model=bert_encoder)
    init_checkpoint = config['bert_model_path']
    ckpt.restore(init_checkpoint).assert_existing_objects_matched()
    # config = models.get_config()
    classifier.train(model)


