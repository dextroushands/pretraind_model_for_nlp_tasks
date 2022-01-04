import tensorflow as tf
import os
import json
import pickle
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import numpy as np
from models.sentence_embedding import SentenceEmbedding
from tasks.classifier import ClassifierTask
from tasks.ner_task import NERTask
from tasks.Itr_pair_task import ItrTask


from predictor.predict_base import BasePredictor
import pandas as pd

class Predictor(BasePredictor):
    '''
    预测类
    '''
    def __init__(self, config):

        self.config = config
        super(Predictor, self).__init__(config)

        self.word_to_index = None
        self.label_to_index = None
        self.word_vectors = None
        self.vocab_size = None

        self.load_vocab()
        #创建模型并加载参数
        self.model = self.create_model()
        self.model = self.load_ckpt_model(self.model, self.config['ckpt_model_path'], self.config['model_name'])


    def create_model(self):
        '''
        创建模型
        :return:
        '''
        model = None
        if self.config['model_name'] == 'bert_classifier':
            model = ClassifierTask(self.config).build_model()

        if self.config['model_name'] == 'bert_ner':
            model = NERTask(self.config).build_model()

        if self.config['model_name'] == 'simbert':
            model = ItrTask(self.config).build_model()

        return model

    def load_vocab(self):
        '''
        加载词典
        :return:
        '''
        # with open(os.path.join(self.config['output_path'], "word_to_index.pkl"), "rb") as f:
        #     self.word_to_index = pickle.load(f)
        with open(os.path.join(self.config['output_path'], "label_to_index.pkl"), "rb") as f:
            self.label_to_index = pickle.load(f)

        # if self.config['use_word2vec']:
        #     if os.path.exists(os.path.join(self.config['output_path'], "word_vectors.npy")):
        #         print("load word_vectors")
        #         self.word_vectors = np.load(os.path.join(self.config['output_path'], "word_vectors.npy"),
        #                                     allow_pickle=True)

    def predict(self, sentence):
        '''
        句子预测
        :param sentence:list
        :return:
        '''
        word_ids, segment_ids, word_mask, sequence_length = [], [], [], []

        _word_ids, _segment_ids, _word_mask, _sequence_length = self.tokenizer.encode(sentence)
        word_ids.append(_word_ids)
        segment_ids.append(_segment_ids)
        word_mask.append(_word_mask)
        sequence_length.append(_sequence_length)
        inputs = dict(
            input_word_ids=word_ids,
            input_mask=word_mask,
            input_type_ids=segment_ids,
        )

        infer_input = {
            "input_word_ids": tf.convert_to_tensor(inputs['input_word_ids']),
            "input_mask": tf.convert_to_tensor(inputs['input_mask']),
            "input_type_ids": tf.convert_to_tensor(inputs['input_type_ids']),
        }
        logits = self.model(infer_input, training=False)
        predictions = self.get_predictions(logits)
        label = self.tokenizer.ids_to_tokens(predictions, self.label_to_index)
        return label

    def sequence_predict(self, sentence):
        '''
        序列标注预测
        :param sentence:
        :return:
        '''
        word_ids, segment_ids, word_mask, sequence_length = [], [], [], []

        _word_ids, _segment_ids, _word_mask, _sequence_length = self.tokenizer.encode(sentence)
        word_ids.append(_word_ids)
        segment_ids.append(_segment_ids)
        word_mask.append(_word_mask)
        sequence_length.append(_sequence_length)
        inputs = dict(
            input_word_ids=word_ids,
            input_mask=word_mask,
            input_type_ids=segment_ids,
        )

        infer_input = {
            "input_word_ids": tf.convert_to_tensor(inputs['input_word_ids']),
            "input_mask": tf.convert_to_tensor(inputs['input_mask']),
            "input_type_ids": tf.convert_to_tensor(inputs['input_type_ids']),
        }
        outputs = self.model(infer_input, training=False)
        # decode_results = outputs.numpy().tolist()
        predictions = self.get_predictions(outputs)[0][1:_sequence_length]

        label = self.tokenizer.ids_to_tokens(predictions, self.label_to_index)
        return label



if __name__=='__main__':
    with open("../model_configs/bert_ner.json", 'r') as fr:
        config = json.load(fr)
    predictor = Predictor(config)
    test_data = pd.read_csv(config['test_data'], error_bad_lines=False, sep='\t')
    inputs = test_data['text_a'].values.tolist()[:10]
    labels = test_data['label'].values.tolist()[:10]
    labels = [str(label) for label in labels]
    predictions = []
    count = 0

    for i,sentence in enumerate(inputs):
        prediction = predictor.sequence_predict(sentence)
        # prediction = predictor.predict(sentence)
        print(prediction)
        if prediction[0] == labels[i]:
            print(sentence)
            count += 1
        predictions.extend(prediction)
    # print(predictions)
    print(count/100)
    print(inputs[5])
