import tensorflow as tf

from official.nlp.keras_nlp.encoders.bert_encoder import BertEncoder

class SentenceEmbedding(tf.keras.Model):
    '''
    句子向量
    '''
    def __init__(self,
                 encoder_network: tf.keras.Model,
                 sequence_length,
                 config = None,
                 **kwargs):
        # self.encoder_network = encoder_network
        self.config = config
        self.sequence_length = sequence_length


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
        if isinstance(sequence_length, int):
            first_layer_outputs = encoder_outputs[0][:, :self.sequence_length, :]
            last_layer_outputs = encoder_outputs[-1][:, :self.sequence_length, :]
            average = (first_layer_outputs + last_layer_outputs) / 2.0
            sentence_embedding = tf.reduce_mean(average, axis=1)
        else:
            sentence_embedding = []
            for seq_len in sequence_length:
                first_layer_outputs = encoder_outputs[0][:, :seq_len, :]
                last_layer_outputs = encoder_outputs[-1][:, :seq_len, :]
                average = (first_layer_outputs + last_layer_outputs) / 2.0
                sentence_embedding.append(tf.reduce_mean(average, axis=1))
        _pooler_layer = tf.keras.layers.Dense(
            units=self.config['pooled_output_size'],
            activation='tanh',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            name='pooler_transform')
        outputs = _pooler_layer(sentence_embedding)

        super(SentenceEmbedding, self).__init__(inputs=inputs, outputs=outputs, **kwargs)