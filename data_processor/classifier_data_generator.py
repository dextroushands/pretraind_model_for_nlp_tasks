from data_processor.embedding import embedding
import numpy as np
import pandas as pd
import pickle
import os


class ClassifierDataGenerator(embedding):
    '''
    生成训练数据
    '''
    def __init__(self, config):
        super(ClassifierDataGenerator, self).__init__(config)
        self.config = config
        self.batch_size = config['batch_size']
        self.load_data()
        self.train_data, self.train_label, self.eval_data, self.eval_label = self.train_eval_split(self.word_ids,
                                                                                                   self.segment_ids,
                                                                                                   self.word_mask,
                                                                                                   self.sequence_length,
                                                                                                   self.labels_idx, 0.2)

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
                                                                                                     np.array(train_data["segment_ids"]),\
                                                                                                     np.array(train_data["word_mask"]),\
                                                                                                     np.array(train_data["sequence_length"]),\
                                                                                                     np.array(train_data["labels_idx"])

            # self.vocab = self.word_to_index.keys()
            # self.vocab_size = len(self.vocab)
        else:
            # 1，读取原始数据
            inputs, labels = self._read_data(self.config['data_path'])
            print("read finished")

            # 选择分词方式
            # if self.config['embedding_type'] == 'char':
            #     all_words = self.cut_chars(inputs)
            # else:
            #     all_words = self.cut_words(inputs)
            # word_to_index = self.word_to_index(all_words)
            label_to_index = self.label_to_index(labels)

            word_ids, segment_ids, word_mask, sequence_length, label_ids = self.save_input_tokens(inputs, labels, label_to_index)
            print('text to tokens process finished')

            # # 2，得到去除低频词和停用词的词汇表
            # word_to_index, all_words = self.word_to_index(inputs)
            # print("word process finished")
            #
            # # 3，得到词汇表
            # label_to_index = self.label_to_index(labels)
            # print("vocab process finished")
            #
            # # 4，输入转索引
            # inputs_idx = [self.tokens_to_ids(text, word_to_index) for text in all_words]
            # print("index transform finished")
            #
            # # 5，对输入做padding
            # inputs_idx = self.padding(inputs_idx)
            # print("padding finished")
            #
            # # 6，标签转索引
            # labels_idx = self.tokens_to_ids(labels, label_to_index)
            # print("label index transform finished")

            # 7, 加载词向量
            # if self.config['word2vec_path']:
            #     word_vectors = self.get_word_vectors(self.vocab)
            #     self.word_vectors = word_vectors
                # 将本项目的词向量保存起来
                # self.save_vectors(self.word_vectors, 'word_vectors')

            # train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
            # with open(os.path.join(self.config['output_path'], "train_data.pkl"), "wb") as fw:
            #     pickle.dump(train_data, fw)
            # labels_idx = labels
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
            batch_output_ids.extend(target_ids)

            if len(batch_word_ids) == self.batch_size:
                yield dict(
                    input_word_ids=np.array(batch_word_ids, dtype="int64"),
                    input_mask=np.array(batch_word_mask, dtype="int64"),
                    input_type_ids=np.array(batch_segment_ids, dtype="int64"),
                    sequence_length=np.array(batch_sequence_length, dtype="int64"),
                    input_target_ids=np.array(batch_output_ids, dtype="float32")
                )
                batch_word_ids, batch_segment_ids, batch_word_mask, batch_sequence_length, batch_output_ids = [], [], [], [], []

