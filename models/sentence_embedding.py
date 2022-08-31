import tensorflow as tf

from official.nlp.keras_nlp.encoders.bert_encoder import BertEncoder

class SentenceEmbedding(tf.keras.Model):
    '''
    句子向量
    '''
    def __init__(self,
                 encoder_network: tf.keras.Model,
                 # sequence_length,
                 config = None,
                 **kwargs):
        # self.encoder_network = encoder_network
        self.config = config
        # self.sequence_length = sequence_length

        # sequence_length = tf.keras.Input(shape=(None,), dtype=tf.int32, name='seqence_length')
        sequence_length = self.config['seq_len']
        inputs = encoder_network.inputs
        outputs = encoder_network(inputs)
        if isinstance(outputs, list):
            sequence_output = outputs[0][-1]
            cls_output = outputs[1]
            encoder_outputs = outputs[0]
        else:
            sequence_output = outputs['sequence_output']
            cls_output = outputs['pooled_output']
            encoder_outputs = outputs['encoder_outputs']

        #取第一层和最后一层的均值作为句子embedding
        # if isinstance(sequence_length, int):
        first_layer_outputs = encoder_outputs[0][:, :sequence_length, :]
        last_layer_outputs = encoder_outputs[-1][:, :sequence_length, :]
        average = (first_layer_outputs + last_layer_outputs) / 2.0
        sentence_embedding = tf.reduce_mean(average, axis=1)
        # else:
        #     sentence_embedding = []
        #     for i in range(self.config['batch_size']):
        #         first_layer_outputs = encoder_outputs[0][:, :sequence_length[i], :]
        #         last_layer_outputs = encoder_outputs[-1][:, :sequence_length[i], :]
        #         average = (first_layer_outputs + last_layer_outputs) / 2.0
        #         sentence_embedding.append(tf.reduce_mean(average, axis=1))
        _pooler_layer = tf.keras.layers.Dense(
            units=self.config['pooled_output_size'],
            activation='tanh',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='pooler_transform')
        outputs = _pooler_layer(sentence_embedding)

        super(SentenceEmbedding, self).__init__(inputs=inputs, outputs=outputs, **kwargs)