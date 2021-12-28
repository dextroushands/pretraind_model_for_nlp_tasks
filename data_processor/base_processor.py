'''
数据预处理的基础对象
'''
import os
import jieba
from collections import Counter
import jieba.posseg as pseg
import pandas as pd

class data_base(object):
    '''
    中文文本处理的基础组件
    '''
    def __init__(self, data_config):
        self.config = data_config


    @staticmethod
    def read_data(path):
        '''
        读取数据集
        :param path:
        :return: text, label
        '''
        texts = []
        labels = []
        with open(path, "rb", encoding='utf8') as f:
            for line in f.readlines():
                text, label = line.strip().split(' ')
                texts.append(text.strip())
                labels.append(label.strip())
        return texts, labels


    @staticmethod
    def _read_data(path):
        """
        读取多标签数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = [[]]
        """
        inputs = []
        labels = []
        train_data = pd.read_csv(path, error_bad_lines=False, sep='\t')
        print(train_data.columns)
        print(train_data.head(2))
        inputs = train_data['text_a'].values.tolist()[:100]
        labels = train_data['label'].values.tolist()[:100]
        labels = [str(label) for label in labels]
        # inputs = [list(i) for i in inputs]

        return inputs, labels

    def get_all_words(self, tokens):
        '''
        对已经分词的数据直接获取所有词
        :param tokens:
        :return:
        '''
        all_words = []
        [all_words.extend(i) for i in tokens]
        return all_words

    def cut_words(self, texts):
        '''
        分词
        :param text:
        :return:
        '''
        all_words = []
        for text in texts:
            words = jieba.lcut(text)
            all_words.extend(words)
        return all_words

    def cut_chars(self, texts):
        '''
        将文本分割成字
        :param text:
        :return:
        '''
        all_chars = []
        for text in texts:
            chars = list(text)
            all_chars.extend(chars)
        return all_chars

    def word_pos_filter(self, pos_filter, text):
        '''
        根据词性过滤文本
        :param pos: ['nr'...]
        :param text:
        :return:
        '''
        words = []
        pos_text = pseg.lcut(text)
        for word, pos in pos_text:
            if pos not in pos_filter:
                words.append(word)
        return words

    def word_freq_filter(self, freq, all_words):
        '''
        词频过滤
        :param freq:
        :return:
        '''
        print(all_words)
        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sort_word_count if item[1] >= freq]
        return words

    def get_vocab(self, all_words):
        '''
        获取词列表
        :param all_words:
        :return:
        '''
        word_count = Counter(all_words)  # 统计词频
        sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        vocab = [item[0] for item in sort_word_count]

        return vocab

    def remove_stop_words(self, all_words):
        '''
        去除停用词
        :param all_words:
        :return:
        '''
        stop_words = self.load_stop_words(self.config['stop_word_path'])
        words = [word for word in all_words if word not in stop_words]
        return words

    def load_stop_words(self, stop_word_path):
        '''
        加载停用词表
        :param stop_word_path:
        :return:
        '''
        with open(stop_word_path, "r", encoding="utf8") as fr:
            stop_words = [line.strip() for line in fr.readlines()]
        return stop_words

