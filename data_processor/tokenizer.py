'''
文本转化成tokens
'''
from data_processor.base_processor import data_base
from itertools import chain
import numpy as np
import pickle
import os
from official.nlp.bert import tokenization


class tokenizer(data_base):
    '''
    文本转tokens
    '''
    def __init__(self, token_configs):
        self.token_configs = token_configs
        super(tokenizer, self).__init__(token_configs)

    def tokens_to_ids(self, tokens, tokens_to_index):
        '''
        token转索引
        :param tokens:
        :return:
        '''
        ids = [tokens_to_index.get(token, 1) for token in tokens]
        return ids

    def labels_to_ids(self, labels, labels_to_index):
        '''
        token转索引
        :param tokens:
        :return:
        '''
        ids = [labels_to_index.get(token) for token in labels]
        return ids

    def seq_labels_to_ids(self, labels, labels_to_index):
        '''
        token转索引
        :param tokens:
        :return:
        '''
        if len(labels) < self.config['seq_len']:
            labels += ['O'] * (self.config['seq_len'] - len(labels))
        else:
            labels = labels[:self.config['seq_len']]
        nan_id = labels_to_index.get('O')
        ids = [labels_to_index.get(token, nan_id) for token in labels]
        return ids

    def seq2seq_label_process(self, labels):
        '''
        seq2seq任务处理label数据，在头尾添加<SOS>,<EOS>
        :param labels:
        :return:
        '''
        res = []
        for line in labels:
            line.insert(0, "<SOS>")
            line.insert(-1, "<EOS>")
            res.append(line)
        return res

    def ids_to_tokens(self, ids, tokens_to_index):
        '''
        索引转成token
        :param ids:
        :return:
        '''
        tokens = [list(tokens_to_index.keys())[id] for id in ids]
        return tokens

    def multi_label_to_index(self, labels, label_to_index):
        '''
        多标签数据转索引
        :param labels:
        :return:
        '''
        label_idxs = np.zeros((len(labels), len(label_to_index)))

        for i, label in enumerate(labels):
            for l in label:
                id = label_to_index.get(l)
                label_idxs[i, id] = 1
        return label_idxs

    def word_to_index(self, all_words):
        '''
        生成词汇-索引字典
        :param texts:
        :return:
        '''

        #是否过滤低频词
        if self.config['freq_filter']:
            vocab = self.word_freq_filter(self.config['freq_filter'], all_words)
        else:
            vocab = self.get_vocab(all_words)
        #设置词典大小
        vocab = ["<PAD>", "<UNK>"] + vocab
        self.vocab_size = self.config['vocab_size']
        if len(vocab) < self.vocab_size:
            self.vocab_size = len(vocab)
        self.vocab = vocab[:self.vocab_size]
        #构建词典索引
        word_to_index = dict(zip(vocab, list(range(len(vocab)))))

        return word_to_index


    def label_to_index(self, labels):
        '''
        标签索引字典
        :param labels:
        :return:
        '''
        if not self.config['multi_label']:
            unique_labels = list(set(labels))  # 单标签转换
        else:
            unique_labels = list(set(chain(*labels)))#多标签转换
        label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return label_to_index

    def padding(self, tokens):
        '''
        将输入序列做定长处理
        :param tokens:
        :return:
        '''
        if len(tokens) < self.config['seq_len']:
            tokens += [0] * (self.config['seq_len'] - len(tokens))
        else:
            tokens = tokens[:self.config['seq_len']]
        return tokens

    def encode(self, text):
        '''
        句子转成token
        :param file_path:
        :return:
        '''
        _tokenizer = tokenization.FullTokenizer(self.config['vocab_path'], do_lower_case=True)
        if isinstance(text, str):

            split_tokens = _tokenizer.tokenize(text)
        else:
            split_tokens = text
        if len(split_tokens) > self.config['seq_len']:
            split_tokens = split_tokens[:self.config['seq_len']]
            sequence_length = self.config['seq_len']
        else:
            sequence_length = len(split_tokens)
            while (len(split_tokens) < self.config['seq_len']):
                split_tokens.append("[PAD]")
            # word_mask = [[1]*(maxlen+2) for i in range(data_len)]

        tokens = []
        tokens.append("[CLS]")
        for i in split_tokens:
            if i not in _tokenizer.vocab:
                tokens.append("[UNK]")
                print(i)
                continue
            tokens.append(i)
        tokens.append("[SEP]")
        word_ids = _tokenizer.convert_tokens_to_ids(tokens)
        word_mask = []
        for i in word_ids:
            if i == "[PAD]":
                word_mask.append(0)
            else:
                word_mask.append(1)
        segment_ids = [0] * len(word_ids)
        return word_ids, segment_ids, word_mask, sequence_length

    def encode_v2(self, text_1, text_2):
        '''
        交互式文本匹配编码
        '''
        _tokenizer = tokenization.FullTokenizer(self.config['vocab_path'], do_lower_case=True)
        if isinstance(text_1, str):
            split_tokens_1 = _tokenizer.tokenize(text_1)
        else:
            split_tokens_1 = text_1
        if isinstance(text_2, str):
            split_tokens_2 = _tokenizer.tokenize(text_2)
        else:
            split_tokens_2 = text_1

        if len(split_tokens_1) + len(split_tokens_2) > self.config['seq_len']:
            split_tokens_2 = split_tokens_2[:self.config['seq_len'] - len(split_tokens_1)]
            sequence_length = self.config['seq_len']
        else:
            sequence_length = len(split_tokens_1) + len(split_tokens_2)
            while (len(split_tokens_1) + len(split_tokens_2) < self.config['seq_len']):
                split_tokens_2.append("[PAD]")

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for i in split_tokens_1:
            if i not in _tokenizer.vocab:
                tokens.append("[UNK]")
                print(i)
                continue
            tokens.append(i)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        for i in split_tokens_2:
            if i not in _tokenizer.vocab:
                tokens.append("[UNK]")
                print(i)
                continue
            tokens.append(i)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        word_ids = _tokenizer.convert_tokens_to_ids(tokens)
        word_mask = []
        for i in word_ids:
            if i == "[PAD]":
                word_mask.append(0)
            else:
                word_mask.append(1)
        return word_ids, segment_ids, word_mask, sequence_length

    def save_input_tokens(self, texts, labels, label_to_index):
        '''
        保存处理完成的输入tokens，方便后续加载
        :param texts:
        :return:
        '''

        word_ids, segment_ids, word_mask, sequence_length = [], [], [], []
        label_ids = []
        for i,text in enumerate(texts):

            _word_ids, _segment_ids, _word_mask, _sequence_length = self.encode(text)
            word_ids.append(_word_ids)
            segment_ids.append(_segment_ids)
            word_mask.append(_word_mask)
            sequence_length.append(_sequence_length)
            label_id = self.labels_to_ids([labels[i]], label_to_index)
            label_ids.append(label_id)


        input_tokens = dict(word_ids=word_ids, segment_ids=segment_ids, word_mask=word_mask, sequence_length=sequence_length, labels_idx=label_ids)
        if not os.path.exists(self.config['output_path']):
            os.mkdir(self.config['output_path'])
        #保存准备训练的tokens数据
        with open(os.path.join(self.config['output_path'], 'train_tokens.pkl'), "wb") as fw:
            pickle.dump(input_tokens, fw)
        # 保存预处理的word_to_index数据
        # with open(os.path.join(self.config['output_path'], 'word_to_index.pkl'), "wb") as fw:
        #     pickle.dump(word_to_index, fw)
        # 保存预处理的word_to_index数据
        with open(os.path.join(self.config['output_path'], 'label_to_index.pkl'), "wb") as fw:
            pickle.dump(label_to_index, fw)
        return word_ids, segment_ids, word_mask, sequence_length, label_ids