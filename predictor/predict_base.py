import tensorflow as tf
import os
import pickle
import json
from data_processor.tokenizer import tokenizer
from keras_models.model_base import BaseModel


class BasePredictor(BaseModel):
    '''
    构建预测的基础对象
    '''
    def __init__(self, config):
        self.tokenizer = tokenizer(config)
        super(BasePredictor, self).__init__(config)

    def load_ckpt_model(self, model, path, model_name):
        '''
        加载ckpt模型
        :param model_path:
        :return:
        '''
        # model = self.create_model()
        path = os.path.join(path, model_name)
        model.load_weights(path)
        return model

    def create_model(self):
        '''
        创建模型
        :return:
        '''
        raise NotImplemented

    def load_vocab(self):
        '''
        加载词典
        :return:
        '''
        raise NotImplemented

    def predict(self, sentence):
        '''
        预测句子结果
        :param sentence:
        :return:
        '''
        raise NotImplemented
