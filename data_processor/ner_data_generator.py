from data_processor.embedding import embedding
import numpy as np
import pandas as pd
import pickle
import os


class NERDataGenerator(embedding):
    '''
    生成训练数据
    '''
    def __init__(self, config):
        super(NERDataGenerator, self).__init__(config)
        self.config = config
        self.batch_size = config['batch_size']
        self.load_data()
        self.train_data, self.train_label, self.eval_data, self.eval_label = self.train_eval_split(self.word_ids,
                                                                                                   self.segment_ids,
                                                                                                   self.word_mask,
                                                                                                   self.sequence_length,
                                                                                                   self.labels_idx, 0.2)

    def read_data(self, path):
        inputs = []
        labels = []
        with open(os.path.join(path, 'source_BIO_2014_cropus.txt'), 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                inputs.append(line.split(sep=' '))
        with open(os.path.join(path, 'target_BIO_2014_cropus.txt'), 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                labels.append(line.split(sep=' '))
        return inputs[:100], labels[:100]

    def get_labels(self):
        return ['O', 'B_LOC', 'I_LOC', 'B_PER', 'I_PER', 'B_ORG', 'I_ORG', 'B_T', 'I_T']

    def save_input_tokens(self, texts, labels, label_to_index):
        '''
        保存处理完成的输入tokens，方便后续加载
        :param texts:
        :return:
        '''
        word_ids, segment_ids, word_mask, sequence_length = [], [], [], []
        label_ids = []
        for i, text in enumerate(texts):
            _word_ids, _segment_ids, _word_mask, _sequence_length = self.encode(text)
            word_ids.append(_word_ids)
            segment_ids.append(_segment_ids)
            word_mask.append(_word_mask)
            sequence_length.append(_sequence_length)
            label_id = self.seq_labels_to_ids(labels[i], label_to_index)
            label_ids.append(label_id)
        input_tokens = dict(word_ids=word_ids, segment_ids=segment_ids, word_mask=word_mask,
                            sequence_length=sequence_length, labels_idx=label_ids)
        if not os.path.exists(self.config['output_path']):
            os.mkdir(self.config['output_path'])
        # 保存准备训练的tokens数据
        with open(os.path.join(self.config['output_path'], 'train_tokens.pkl'), "wb") as fw:
            pickle.dump(input_tokens, fw)
        # 保存预处理的word_to_index数据
        # with open(os.path.join(self.config['output_path'], 'word_to_index.pkl'), "wb") as fw:
        #     pickle.dump(word_to_index, fw)
        # 保存预处理的word_to_index数据
        with open(os.path.join(self.config['output_path'], 'label_to_index.pkl'), "wb") as fw:
            pickle.dump(label_to_index, fw)
        return word_ids, segment_ids, word_mask, sequence_length, label_ids

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

            if os.path.exists(os.path.join(self.config['output_path'], "word_vectors.npy")):
                print("load word_vectors")
                self.word_vectors = np.load(os.path.join(self.config['output_path'], "word_vectors.npy"),
                                            allow_pickle=True)

            self.word_ids, self.segment_ids, self.word_mask, self.sequence_length, self.labels_idx = np.array(train_data["word_ids"]), \
                                                                                                     np.array(train_data["segment_ids"]), \
                                                                                                     np.array(train_data["word_mask"]), \
                                                                                                     np.array(train_data["sequence_length"]), \
                                                                                                     np.array(train_data["labels_idx"])

            # self.vocab = self.word_to_index.keys()
            # self.vocab_size = len(self.vocab)
        else:
            # 1，读取原始数据
            inputs, labels = self.read_data(self.config['data_path'])
            print("read finished")
            targets = self.get_labels()
            label_to_index = self.label_to_index(targets)

            word_ids, segment_ids, word_mask, sequence_length, label_ids = self.save_input_tokens(inputs, labels,
                                                                                                  label_to_index)
            print('text to tokens process finished')


            self.word_ids, self.segment_ids, self.word_mask, self.sequence_length, self.labels_idx = word_ids, segment_ids, word_mask, sequence_length, label_ids

    def train_eval_split(self, word_ids, segment_ids, word_mask, sequence_length, labels, rate):
        '''
        划分训练和验证集
        :param data:
        :param labels:
        :param rate:
        :return:
        '''
        # np.random.shuffle(data)
        perm = int(len(word_ids) * rate)
        train_data = (word_ids[perm:], segment_ids[perm:], word_mask[perm:], sequence_length[perm:])
        eval_data = (word_ids[:perm], segment_ids[:perm], word_mask[:perm], sequence_length[:perm])
        train_label = labels[perm:]
        eval_label = labels[:perm]
        return train_data, train_label, eval_data, eval_label


    def gen_data(self, input_idx, labels_idx):
        '''
        生成批次数据
        :return:
        '''
        word_ids, segment_ids, word_mask, sequence_length = input_idx[0], input_idx[1], input_idx[2], input_idx[3]
        batch_word_ids, batch_segment_ids, batch_word_mask, batch_sequence_length, batch_output_ids = [], [], [], [], []

        for i in range(len(word_ids)):
            word_id = word_ids[i]
            segment_id = segment_ids[i]
            mask = word_mask[i]
            seq_len = sequence_length[i]
            target_ids = labels_idx[i]
            batch_word_ids.append(word_id)
            batch_segment_ids.append(segment_id)
            batch_word_mask.append(mask)
            batch_sequence_length.append(seq_len)
            batch_output_ids.append(target_ids)

            if len(batch_word_ids) == self.batch_size:
                yield dict(
                    input_word_ids=np.array(batch_word_ids, dtype="int64"),
                    input_mask=np.array(batch_word_mask, dtype="int64"),
                    input_type_ids=np.array(batch_segment_ids, dtype="int64"),
                    sequence_length=np.array(batch_sequence_length, dtype="int64"),
                    input_target_ids=np.array(batch_output_ids, dtype="float32")
                )
                batch_word_ids, batch_segment_ids, batch_word_mask, batch_sequence_length, batch_output_ids = [], [], [], [], []

