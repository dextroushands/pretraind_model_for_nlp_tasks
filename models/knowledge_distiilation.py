import tensorflow as tf
from official.nlp.modeling import layers
from official.nlp.modeling import networks


class Distill_model(tf.keras.Model):
    '''
    使用dssm进行知识蒸馏
    '''
    def __init__(self,
                 config,
                 teacher_network,
                 vocab_size,
                 word_vectors,
                 **kwargs):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors
        #冻结teacher network的参数
        for layer in teacher_network.layers:
            layer.trainable = False
        #定义学生模型输入
        query = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_x_ids')
        sim_query = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_y_ids')
        #定义老师模型输入
        word_ids_a = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids_a')
        mask_a = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask_a')
        type_ids_a = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids_a')
        word_ids_b = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids_b')
        mask_b = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask_b')
        type_ids_b = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids_b')
        input_a = [word_ids_a, mask_a, type_ids_a]
        input_b = [word_ids_b, mask_b, type_ids_b]
        teacher_input = [input_a, input_b]

        #teacher_softlabel
        teacher_output = teacher_network(teacher_input)
        # teacher_classifier = layers.ClassificationHead(
        #   inner_dim=teacher_output.shape[-1],
        #   num_classes=self.config['num_classes'],
        #   dropout_rate=self.config['dropout_rate'],
        #   name='sentence_prediction')
        # teacher_logits = teacher_classifier(teacher_output)
        teacher_soft_label = softmax_t(self.config['t'], teacher_output['logits'])

        # embedding层
        # 利用词嵌入矩阵将输入数据转成词向量，shape=[batch_size, seq_len, embedding_size]
        class GatherLayer(tf.keras.layers.Layer):
            def __init__(self, config, vocab_size, word_vectors):
                super(GatherLayer, self).__init__()
                self.config = config

                self.vocab_size = vocab_size
                self.word_vectors = word_vectors

            def build(self, input_shape):
                with tf.name_scope('embedding'):
                    if not self.config['use_word2vec']:
                        self.embedding_w = tf.Variable(tf.keras.initializers.glorot_normal()(
                            shape=[self.vocab_size, self.config['embedding_size']],
                            dtype=tf.float32), trainable=True, name='embedding_w')
                    else:
                        self.embedding_w = tf.Variable(tf.cast(self.word_vectors, tf.float32), trainable=True,
                                                       name='embedding_w')
                self.build = True

            def call(self, inputs, **kwargs):
                return tf.gather(self.embedding_w, inputs, name='embedded_words')

            def get_config(self):
                config = super(GatherLayer, self).get_config()

                return config

        # class shared_net(tf.keras.Model):
        #     def __init__(self, config, vocab_size, word_vectors):
        #         query = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_x_ids')
        #         query_embedding = GatherLayer(config, vocab_size, word_vectors)(query)
        #         query_embedding_output = shared_lstm_layer(config)(query_embedding)
        #
        #         super(shared_net, self).__init__(inputs=[query], outputs=query_embedding_output)
        shared_net = tf.keras.Sequential([GatherLayer(config, vocab_size, word_vectors),
                                          shared_lstm_layer(config)])

        query_embedding_output = shared_net.predict_step(query)
        sim_query_embedding_output = shared_net.predict_step(sim_query)


        # 余弦函数计算相似度
        # cos_similarity余弦相似度[batch_size, similarity]
        query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_embedding_output), axis=-1), name='query_norm')
        sim_query_norm = tf.sqrt(tf.reduce_sum(tf.square(sim_query_embedding_output), axis=-1), name='sim_query_norm')

        dot = tf.reduce_sum(tf.multiply(query_embedding_output, sim_query_embedding_output), axis=-1)
        cos_similarity = tf.divide(dot, (query_norm * sim_query_norm), name='cos_similarity')
        self.similarity = cos_similarity

        # 预测为正例的概率
        cond = (self.similarity > self.config["neg_threshold"])
        pos = tf.where(cond, tf.square(self.similarity), 1 - tf.square(self.similarity))
        neg = tf.where(cond, 1 - tf.square(self.similarity), tf.square(self.similarity))
        predictions = [[neg[i], pos[i]] for i in range(self.config['batch_size'])]

        self.logits = self.similarity
        student_soft_label = softmax_t(self.config['t'], self.logits)
        student_hard_label = self.logits
        if self.config['is_training']:
            #训练时候蒸馏
            outputs = dict(student_soft_label=student_soft_label, student_hard_label=student_hard_label, teacher_soft_label=teacher_soft_label, predictions=predictions)
            super(Distill_model, self).__init__(inputs=[query, sim_query, teacher_input], outputs=outputs, **kwargs)
        else:
            #预测时候只加载学生模型
            outputs = dict(predictions=predictions)
            super(Distill_model, self).__init__(inputs=[query, sim_query], outputs=outputs, **kwargs)



def softmax_t(t, logits):
    '''
    带参数t的softmax
    '''
    _sum = tf.reduce_sum(tf.exp(logits/t))
    return tf.exp(logits/t) / _sum

class shared_lstm_layer(tf.keras.layers.Layer):
    '''
    共享lstm层参数
    '''
    def __init__(self, config):
        self.config = config
        super(shared_lstm_layer, self).__init__()

    def build(self, input_shape):
        forward_layer_1 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                               return_sequences=True)
        backward_layer_1 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                                return_sequences=True, go_backwards=True)
        forward_layer_2 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                               return_sequences=True)
        backward_layer_2 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                                return_sequences=True, go_backwards=True)
        self.bilstm_1 = tf.keras.layers.Bidirectional(forward_layer_1, backward_layer=backward_layer_1)
        self.bilstm_2 = tf.keras.layers.Bidirectional(forward_layer_2, backward_layer=backward_layer_2)
        self.layer_dropout = tf.keras.layers.Dropout(0.4)
        self.output_dense = tf.keras.layers.Dense(self.config['output_size'])

        super(shared_lstm_layer, self).build(input_shape)

    def get_config(self):
        config = {}
        return config

    def call(self, inputs, **kwargs):
        query_res_1 = self.bilstm_1(inputs)
        query_res_1 = self.layer_dropout(query_res_1)
        query_res_2 = self.bilstm_2(query_res_1)

        #取时间步的平均值，摊平[batch_size, forward_size+backward_size]
        avg_query_embedding = tf.reduce_mean(query_res_2, axis=1)
        tmp_query_embedding = tf.reshape(avg_query_embedding, [self.config['batch_size'], self.config['hidden_size']*2])
        # 全连接层[batch_size, dense_dim]
        query_embedding_output = self.output_dense(tmp_query_embedding)
        query_embedding_output = tf.keras.activations.relu(query_embedding_output)
        return query_embedding_output