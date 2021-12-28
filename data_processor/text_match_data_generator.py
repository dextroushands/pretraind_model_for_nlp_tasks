from data_processor.embedding import embedding
import numpy as np
import pandas as pd
import pickle
import os
from random import shuffle

class TextMatchDataGenerator(embedding):
    '''
    生成训练数据
    '''
    def __init__(self, config):
        super(TextMatchDataGenerator, self).__init__(config)
        self.config = config
        self.batch_size = config['batch_size']
        self.load_data()
        self.train_data, self.train_label, self.eval_data, self.eval_label = self.train_eval_split(self.query_word_idx, self.query_segment_idx, self.query_word_mask, self.query_sequence_length, \
            self.sim_word_idx, self.sim_segment_idx, self.sim_word_mask, self.sim_sequence_length, self.labels_idx, 0.2)

    def read_data(self, file_path, data_size=100):
        '''
        加载训练数据
        '''
        df = pd.read_csv(file_path)
        # query = [jieba.lcut(i) for i in df['sentence1'].values[0:data_size]]
        # sim = [jieba.lcut(i) for i in df['sentence2'].values[0:data_size]]
        query = [list(i) for i in df['sentence1'].values[0:data_size]]
        sim = [list(i) for i in df['sentence2'].values[0:data_size]]
        label = df['label'].values[0:data_size]

        return query, sim, label

    def save_input_tokens(self, query, sim, labels, label_to_index):
        '''
        保存处理完成的输入tokens，方便后续加载
        :param texts:
        :return:
        '''

        query_word_ids, query_segment_ids, query_word_mask, query_sequence_length = [], [], [], []
        sim_word_ids, sim_segment_ids, sim_word_mask, sim_sequence_length = [], [], [], []

        label_ids = []
        for i in range(len(query)):
            _query_word_ids, _query_segment_ids, _query_word_mask, _query_sequence_length = self.encode(query[i])
            _sim_word_ids, _sim_segment_ids, _sim_word_mask, _sim_sequence_length = self.encode(sim[i])

            query_word_ids.append(_query_word_ids)
            query_segment_ids.append(_query_segment_ids)
            query_word_mask.append(_query_word_mask)
            query_sequence_length.append(_query_sequence_length)

            sim_word_ids.append(_sim_word_ids)
            sim_segment_ids.append(_sim_segment_ids)
            sim_word_mask.append(_sim_word_mask)
            sim_sequence_length.append(_sim_sequence_length)

            label_id = self.labels_to_ids([labels[i]], label_to_index)
            label_ids.append(label_id)
        input_tokens = dict(query_word_ids=query_word_ids, query_segment_ids=query_segment_ids, query_word_mask=query_word_mask,
                            query_sequence_length=query_sequence_length,sim_word_ids=sim_word_ids,
                            sim_segment_ids=sim_segment_ids, sim_word_mask=sim_word_mask,
                            sim_sequence_length=sim_sequence_length,labels_idx=label_ids)
        if not os.path.exists(self.config['output_path']):
            os.mkdir(self.config['output_path'])
        #保存准备训练的tokens数据
        with open(os.path.join(self.config['output_path'], 'train_tokens.pkl'), "wb") as fw:
            pickle.dump(input_tokens, fw)
        # 保存预处理的label_to_index数据
        with open(os.path.join(self.config['output_path'], 'label_to_index.pkl'), "wb") as fw:
            pickle.dump(label_to_index, fw)
        return query_word_ids, query_segment_ids, query_word_mask, query_sequence_length,\
               sim_word_ids, sim_segment_ids, sim_word_mask, sim_sequence_length, label_ids

    def load_data(self):
        '''
        加载预处理好的数据
        :return:
        '''

        if os.path.exists(os.path.join(self.config['output_path'], "train_tokens.pkl")) and \
                os.path.exists(os.path.join(self.config['output_path'], "label_to_index.pkl")):
            print("load existed train data")
            # with open(os.path.join(self.config['output_path'], "word_to_index.pkl"), "rb") as f:
            #     self.word_to_index = pickle.load(f)
            with open(os.path.join(self.config['output_path'], "label_to_index.pkl"), "rb") as f:
                self.label_to_index = pickle.load(f)
            with open(os.path.join(self.config['output_path'], "train_tokens.pkl"), "rb") as f:
                train_data = pickle.load(f)

            self.query_word_idx, self.query_segment_idx, self.query_word_mask, self.query_sequence_length, \
            self.sim_word_idx, self.sim_segment_idx, self.sim_word_mask, self.sim_sequence_length, self.labels_idx = np.array(train_data["query_word_ids"]), \
                                                                                                     np.array(train_data["query_segment_ids"]), \
                                                                                                     np.array(train_data["query_word_mask"]), \
                                                                                                     np.array(train_data["query_sequence_length"]), \
                                                                                                     np.array(train_data["sim_word_ids"]), \
                                                                                                     np.array(train_data["sim_segment_ids"]), \
                                                                                                     np.array(train_data["sim_word_mask"]), \
                                                                                                     np.array(train_data["sim_sequence_length"]), \
                                                                                                     np.array(train_data["labels_idx"])
        else:
            # 1，读取原始数据
            query, sim, labels = self.read_data(self.config['data_path'])
            print("read finished")

            label_to_index = self.label_to_index(labels)

            query_word_ids, query_segment_ids, query_word_mask, query_sequence_length, \
            sim_word_ids, sim_segment_ids, sim_word_mask, sim_sequence_length, label_ids = self.save_input_tokens(query, sim, labels, label_to_index)
            print('text to tokens process finished')

            # train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
            # with open(os.path.join(self.config['output_path'], "train_data.pkl"), "wb") as fw:
            #     pickle.dump(train_data, fw)
            # labels_idx = labels
            self.query_word_idx, self.query_segment_idx, self.query_word_mask, self.query_sequence_length, \
            self.sim_word_idx, self.sim_segment_idx, self.sim_word_mask, self.sim_sequence_length,self.labels_idx = query_word_ids, query_segment_ids, query_word_mask, query_sequence_length,\
               sim_word_ids, sim_segment_ids, sim_word_mask, sim_sequence_length, label_ids

    def train_eval_split(self, query_word_ids, query_segment_ids, query_word_mask, query_sequence_length,
                         sim_word_ids, sim_segment_ids, sim_word_mask, sim_sequence_length, labels, rate):

        split_index = int(len(query_word_ids) * rate)
        train_data = (query_word_ids[split_index:], query_segment_ids[split_index:], query_word_mask[split_index:],
                      query_sequence_length[split_index:], sim_word_ids[split_index:], sim_segment_ids[split_index:],
                      sim_word_mask[split_index:], sim_sequence_length[split_index:])
        train_label = labels[split_index:]
        eval_data = (query_word_ids[:split_index], query_segment_ids[:split_index], query_word_mask[:split_index],
                      query_sequence_length[:split_index], sim_word_ids[:split_index], sim_segment_ids[:split_index],
                      sim_word_mask[:split_index], sim_sequence_length[:split_index])
        eval_label = labels[:split_index]

        return train_data, train_label, eval_data, eval_label

    def gen_data(self, inputs_idx, labels_idx):
        '''
        生成批次数据
        :return:
        '''
        query_word_ids, query_segment_ids, query_word_mask, query_sequence_length, \
        sim_word_ids, sim_segment_ids, sim_word_mask, sim_sequence_length = inputs_idx[0], inputs_idx[1],inputs_idx[2],\
                                                                                       inputs_idx[3],inputs_idx[4],inputs_idx[5],\
                                                                                       inputs_idx[6],inputs_idx[7]
        batch_word_ids_a, batch_segment_ids_a, batch_word_mask_a, batch_sequence_length_a, \
        batch_word_ids_b, batch_segment_ids_b, batch_word_mask_b, batch_sequence_length_b, batch_output_ids= [], [], [], [], [], [], [], [], []

        for i in range(len(query_word_ids)):
            batch_word_ids_a.append(query_word_ids[i])
            batch_segment_ids_a.append(query_segment_ids[i])
            batch_word_mask_a.append(query_word_mask[i])
            batch_sequence_length_a.append(query_sequence_length[i])

            batch_word_ids_b.append(sim_word_ids[i])
            batch_segment_ids_b.append(sim_segment_ids[i])
            batch_word_mask_b.append(sim_word_mask[i])
            batch_sequence_length_b.append(sim_sequence_length[i])

            batch_output_ids.append(labels_idx[i])


            if len(batch_output_ids) == self.batch_size:
                yield dict(
                input_word_ids_a=np.array(batch_word_ids_a, dtype="int32"),
                input_mask_a=np.array(batch_word_mask_a, dtype="int32"),
                input_type_ids_a=np.array(batch_segment_ids_a, dtype="int32"),
                input_word_ids_b=np.array(batch_word_ids_b, dtype="int32"),
                input_mask_b=np.array(batch_word_mask_b, dtype="int32"),
                input_type_ids_b=np.array(batch_segment_ids_b, dtype="int32"),
                input_target_ids=np.array(batch_output_ids, dtype="float32")
                )
                batch_word_ids_a, batch_segment_ids_a, batch_word_mask_a, batch_sequence_length_a, \
                batch_word_ids_b, batch_segment_ids_b, batch_word_mask_b, batch_sequence_length_b, batch_output_ids = [], [], [], [], [], [], [], [], []

