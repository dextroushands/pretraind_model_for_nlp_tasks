import os
import tensorflow as tf
import functools
from typing import Any, Callable, Optional
import abc


class BaseModel(tf.Module, metaclass=abc.ABCMeta):
    '''
    模型的基类
    '''
    def __init__(self, config):
        self.config = config
        super(BaseModel, self).__init__()

    def build_model(self):
        '''
        创建模型
        :return:
        '''
        raise NotImplemented

    def build_inputs(self, inputs):
        '''
        创建输入
        :return:
        '''
        raise NotImplemented

    def build_losses(self, labels, model_outputs, metrics, aux_losses) -> tf.Tensor:
        '''
        计算loss值
        :param labels:
        :param model_outputs:
        :param metrics:
        :return:
        '''
        raise NotImplemented

    def build_metrics(self, training: bool = True):
        """
        获取模型训练/验证的评价指标
        :param training:
        :return:
        """
        del training
        return []

    def compile_model(self,
                      model: tf.keras.Model,
                      optimizer: tf.keras.optimizers.Optimizer,
                      loss=None,
                      train_step: Optional[Callable[..., Any]] = None,
                      validation_step: Optional[Callable[..., Any]] = None,
                      **kwargs) -> tf.keras.Model:
        """Compiles the model with objects created by the task.

        The method should not be used in any customized training implementation.

        Args:
          model: a keras.Model.
          optimizer: the keras optimizer.
          loss: a callable/list of losses.
          train_step: optional train step function defined by the task.
          validation_step: optional validation_step step function defined by the
            task.
          **kwargs: other kwargs consumed by keras.Model compile().

        Returns:
          a compiled keras.Model.
        """
        if bool(loss is None) == bool(train_step is None):
            raise ValueError("`loss` and `train_step` should be exclusive to "
                             "each other.")
        model.compile(optimizer=optimizer, loss=loss, **kwargs)

        if train_step:
            model.train_step = functools.partial(
                train_step, model=model, optimizer=model.optimizer)
        if validation_step:
            model.test_step = functools.partial(validation_step, model=model)
        return model

    def process_metrics(self, metrics, labels, model_outputs):
        '''
        处理并更新评价指标
        :param metrics:
        :param labels:
        :param model_outputs:
        :return:
        '''
        for metric in metrics:
            metric.update_state(labels, model_outputs)

    def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
        '''
        处理并更新compiled metrics
        :param compiled_metrics:
        :param labels:
        :param model_outputs:
        :return:
        '''
        compiled_metrics.update_state(labels, model_outputs)

    def get_optimizer(self):
        '''
        选择优化算法
        :return:
        '''
        option = self.config['optimizer']
        optimizer = None
        learning_rate = self.config['learning_rate']
        if option == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        if option == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        if option == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate)
        return optimizer

    def train_step(self,
                   inputs,
                   model:tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   metrics=None):
        '''
        进行训练，前向和后向计算
        :param inputs:
        :param model:
        :param optimizer:
        :param metrics:
        :return:
        '''
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = self.build_losses(labels=inputs['labels'], model_outputs=outputs, metrics=metrics, aux_losses=None)
        tvars = model.trainable_variables
        grads = tape.gradient(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
        optimizer.apply_gradients(list(zip(grads, tvars)))
        labels = inputs['labels']
        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
            logs.update({m.name: m.result() for m in model.metrics})
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs


    def validation_step(self, inputs, model:tf.keras.Model, metrics=None):
        '''
        验证集验证模型
        :param input:
        :param model:
        :return:
        '''
        labels = inputs['labels']
        outputs = self.inference_step(inputs, model)
        loss = self.build_losses(labels, outputs, metrics, aux_losses=model.losses)

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def inference_step(self, inputs, model:tf.keras.Model):
        '''
        模型推理
        :param inputs:
        :param model:
        :return:
        '''
        return model(inputs, training=False)

    def get_predictions(self, logits):
        '''
        模型预测结果
        :param input:
        :param model:
        :return:
        '''

        predictions = tf.keras.layers.Activation(
            tf.nn.log_softmax, dtype=tf.float32)(logits).numpy()
        return predictions

    def save_ckpt_model(self, model:tf.keras.Model):
        '''
        将模型保存成ckpt格式
        :param model:
        :return:
        '''
        save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 self.config["ckpt_model_path"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_save_path = os.path.join(save_path, self.config["model_name"])

        # checkpoint = tf.train.Checkpoint(model)
        # checkpoint.save(model_save_path + '/model.ckpt')
        model.save_weights(model_save_path)

    def save_pb_model(self, model:tf.keras.Model, checkpoint_dir=None, restore_model_using_load_weights=True):
        '''
        将模型保存成pb格式
        :param model:
        :return:
        '''
        save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 self.config["export_model_path"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_export_path = os.path.join(save_path, self.config["model_name"])

        if checkpoint_dir:
            # Keras compile/fit() was used to save checkpoint using
            # model.save_weights().
            if restore_model_using_load_weights:
                model_weight_path = os.path.join(checkpoint_dir, 'checkpoint')
                assert tf.io.gfile.exists(model_weight_path)
                model.load_weights(model_weight_path)

            # tf.train.Checkpoint API was used via custom training loop logic.
            else:
                checkpoint = tf.train.Checkpoint(model=model)

                # Restores the model from latest checkpoint.
                latest_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                assert latest_checkpoint_file

                checkpoint.restore(
                    latest_checkpoint_file).assert_existing_objects_matched()

        model.save(model_export_path, include_optimizer=False, save_format='tf')

    def load_ckpt_model(self, model, path, model_name):
        '''
        加载ckpt模型
        :param model_path:
        :return:
        '''
        # model = self.create_model()
        path = os.path.join(path, model_name)
        model.load_weights(path)
        return model









