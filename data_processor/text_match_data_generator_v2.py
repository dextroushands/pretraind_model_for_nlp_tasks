from data_processor.embedding import embedding
import numpy as np
import pandas as pd
import pickle
import os
from random import shuffle
import random
import copy
from itertools import chain

class TextMatchDataGeneratorV2(embedding):
    '''
    生成训练数据
    '''
    def __init__(self, config):
        super(TextMatchDataGeneratorV2, self).__init__(config)
        self.config = config
        self.batch_size = config['batch_size']
        self.load_data()
        self.train_data, self.train_label, self.eval_data, self.eval_label = self.train_eval_split(self.word_idx, self.segment_idx, self.word_mask, self.sequence_length,self.labels_idx, 0.2)

    def read_data(self, file_path):
        '''
        加载训练数据
        '''
        # df = pd.read_csv(file_path)
        # query = [jieba.lcut(i) for i in df['sentence1'].values[0:data_size]]
        # sim = [jieba.lcut(i) for i in df['sentence2'].values[0:data_size]]
        # query = [list(i) for i in df['sentence1'].values]
        # sim = [list(i) for i in df['sentence2'].values]
        # import pandas as pd
        work_data = pd.read_excel(file_path)
        std_query_list = work_data['standard_questions'].tolist()
        sim_query_list = work_data['sim_questions'].tolist()
        # std_answer_list = work_data['standard_answers'].tolist()
        sim = []

        for i in range(len(std_query_list)):
            _sim = sim_query_list[i].split('||')
            sim.append(_sim)


        return std_query_list, sim

    def negative_sampling(self, queries, sim):
        '''
        随机负采样
        '''
        new_queries = []
        labels = []
        for i, item in enumerate(queries):
            copy_questions = copy.copy(queries)
            copy_questions.remove(item)
            neg_samples = random.sample(copy_questions, 5)
            pos_samples = random.sample(sim[i], 2)
            new_queries.append([item] + pos_samples + neg_samples)
            labels.append([1]*2 + [0]*5)
        return new_queries, labels

    def save_ranking_tokens(self, queries, sim):
        '''
        保存处理完成的输入tokens，方便后续加载
        :param texts:
        :return:
        '''

        word_ids, segment_ids, word_mask, sequence_length = [], [], [], []
        word_ids_list, segment_ids_list, word_mask_list, sequence_length_list = [], [], [], []
        new_queries, label_ids = self.negative_sampling(queries, sim)

        for j, questions in enumerate(new_queries):
            for i, query in enumerate(questions[1:]):

                _word_ids, _segment_ids, _word_mask, _sequence_length = self.encode_v2(query[0], query)

                word_ids.append(_word_ids)
                segment_ids.append(_segment_ids)
                word_mask.append(_word_mask)
                sequence_length.append(_sequence_length)

            word_ids_list.append(word_ids)
            segment_ids_list.append(segment_ids)
            word_mask_list.append(word_mask)
            sequence_length_list.append(sequence_length)


            # label_id = self.labels_to_ids([labels[i]], label_to_index)
            # label_ids_list.append(label_ids)
        input_tokens = dict(word_ids=word_ids_list, query_segment_ids=segment_ids_list, query_word_mask=word_mask_list,
                            sequence_length=sequence_length_list,labels_idx=label_ids)
        if not os.path.exists(self.config['output_path']):
            os.mkdir(self.config['output_path'])
        #保存准备训练的tokens数据
        with open(os.path.join(self.config['output_path'], 'train_tokens.pkl'), "wb") as fw:
            pickle.dump(input_tokens, fw)
        # 保存预处理的label_to_index数据
        # with open(os.path.join(self.config['output_path'], 'label_to_index.pkl'), "wb") as fw:
        #     pickle.dump(label_to_index, fw)
        return word_ids_list, segment_ids_list, word_mask_list, sequence_length_list, label_ids

    def load_data(self):
        '''
        加载预处理好的数据
        :return:
        '''

        if os.path.exists(os.path.join(self.config['output_path'], "train_tokens.pkl")) or \
                os.path.exists(os.path.join(self.config['output_path'], "label_to_index.pkl")):
            print("load existed train data")
            # with open(os.path.join(self.config['output_path'], "word_to_index.pkl"), "rb") as f:
            #     self.word_to_index = pickle.load(f)
            # with open(os.path.join(self.config['output_path'], "label_to_index.pkl"), "rb") as f:
            #     self.label_to_index = pickle.load(f)
            with open(os.path.join(self.config['output_path'], "train_tokens.pkl"), "rb") as f:
                train_data = pickle.load(f)

            self.word_idx, self.segment_idx, self.word_mask, self.sequence_length, \
            self.labels_idx = np.array(train_data["word_ids"]), \
                              np.array(train_data["query_segment_ids"]), \
                              np.array(train_data["query_word_mask"]), \
                              np.array(train_data["sequence_length"]), \
                              np.array(train_data["labels_idx"])
        else:
            # 1，读取原始数据
            query, sim = self.read_data(self.config['data_path'])
            print("read finished")

            # label_to_index = self.label_to_index(labels)

            word_ids, segment_ids, word_mask, sequence_length, label_ids = self.save_ranking_tokens(query, sim)
            print('text to tokens process finished')

            # train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
            # with open(os.path.join(self.config['output_path'], "train_data.pkl"), "wb") as fw:
            #     pickle.dump(train_data, fw)
            # labels_idx = labels
            self.word_idx, self.segment_idx, self.word_mask, self.sequence_length, \
            self.labels_idx = word_ids, segment_ids, word_mask, sequence_length, label_ids

    def train_eval_split(self, word_ids, segment_ids, word_mask, sequence_length,
                         labels, rate):

        split_index = int(len(word_ids) * rate)
        train_data = (word_ids[split_index:], segment_ids[split_index:], word_mask[split_index:],
                      sequence_length[split_index:])
        train_label = labels[split_index:]
        eval_data = (word_ids[:split_index], segment_ids[:split_index], word_mask[:split_index],
                      sequence_length[:split_index])
        eval_label = labels[:split_index]

        return train_data, train_label, eval_data, eval_label

    def gen_data(self, inputs_idx, labels_idx):
        '''
        生成批次数据
        :return:
        '''
        word_ids, segment_ids, word_mask, sequence_length = inputs_idx[0], inputs_idx[1],inputs_idx[2],inputs_idx[3]
        batch_word_ids, batch_segment_ids, batch_word_mask, batch_sequence_length, batch_output_ids= [], [], [], [], []

        for i in range(len(word_ids)):
            batch_word_ids.append(word_ids[i])
            batch_segment_ids.append(segment_ids[i])
            batch_word_mask.append(word_mask[i])
            batch_sequence_length.append(sequence_length[i])

            batch_output_ids.append(labels_idx[i])


            if len(batch_output_ids) == self.batch_size:
                yield dict(
                input_word_ids=np.array(list(chain(*batch_word_ids)), dtype="int32"),
                input_mask=np.array(list(chain(*batch_word_mask)), dtype="int32"),
                input_type_ids=np.array(list(chain(*batch_segment_ids)), dtype="int32"),
                input_target_ids=np.array(list(chain(*batch_output_ids)), dtype="float32")
                )
                batch_word_ids, batch_segment_ids, batch_word_mask, batch_sequence_length, batch_output_ids = [], [], [], [], []

