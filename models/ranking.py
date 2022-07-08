import tensorflow as tf
from official.nlp.modeling import layers
from official.nlp.modeling import networks


class Ranking(tf.keras.Model):
    '''
    bert的排序模型
    '''
    def __init__(self, config, network, **kwargs):
        self.config = config
        # 定义模型输入
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids')
        mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_mask')
        type_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_type_ids')
        input = [word_ids, mask, type_ids]

        _output = network(input)
        classifier = networks.Classification(
            input_width=_output[1].shape[-1],
            num_classes=self.config['num_classes'],
            output='logits',
            name='sentence_prediction')
        _logits = classifier(_output[1]) #[batch_size*samples_num, 1]
        logits = tf.split(_logits, num_or_size_splits=self.config['batch_size'], axis=0)
        _relations = tf.keras.layers.Activation(tf.nn.sigmoid)(logits)
        predictions = tf.reshape(tf.argmax(_relations), [-1])
        outputs = dict(logits=logits, predictions=predictions)
        super(Ranking, self).__init__(inputs=input, outputs=outputs, **kwargs)
