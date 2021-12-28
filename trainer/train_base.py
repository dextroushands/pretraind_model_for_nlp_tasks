import tensorflow as tf

from model.model_base import BaseModel


class TrainBase(BaseModel):
    '''
    模型训练基础
    '''
    def __init__(self, train_config):
        self.epoches = train_config['epoches']
        self.data_generator = None

        super(TrainBase, self).__init__(train_config)

    def train(self, model):
        '''
        训练过程
        :return:
        '''
        model.summary()
        optimizer = self.get_optimizer()
        metrics = self.build_metrics()
        batch_num = 0
        valid_loss = 0
        best_acc = 0
        mean_acc = 0
        for i in range(self.epoches):
            print("------------start train epoch {}--------------------".format(i))
            for train_batch in self.data_generator.gen_data(self.data_generator.train_data, self.data_generator.train_label):
                train_input = self.build_inputs(train_batch)
                train_loss = self.train_step(train_input, model, optimizer, metrics)
                print(train_loss)
                batch_num += 1

                if batch_num % 3 == 0:
                    print("------------start validation epoch {}--------------".format(i))
                    count = 0
                    sum_acc = 0
                    for valid_batch in self.data_generator.gen_data(self.data_generator.eval_data, self.data_generator.eval_label):
                        count += 1
                        valid_input = self.build_inputs(valid_batch)
                        valid_loss = self.validation_step(valid_input, model, metrics=metrics)
                        print("accuracy: {}".format(metrics[0].result().numpy())+'\n')
                        sum_acc += metrics[0].result().numpy()
                    mean_acc = sum_acc/count
            if mean_acc > best_acc:
                best_acc = mean_acc
                # print('save model')
                self.save_ckpt_model(model)
                self.save_pb_model(model)

    def fit_train(self, model):
        '''
        使用fit训练模型
        :return:
        '''
        optimizer = self.get_optimizer()
        metrics = self.build_metrics()

        model = self.compile_model(model,
                             optimizer=optimizer,
                             train_step=self.train_step,
                             validation_step=self.validation_step,
                             metrics=metrics)
        model.summary()
        dataset = self.data_generator.gen_data(self.data_generator.train_data, self.data_generator.train_label)
        valid_data = self.data_generator.gen_data(self.data_generator.eval_data, self.data_generator.eval_label)
        # dataset = dataset.repeat()
        # valid_data = valid_data.repeat()
        # dataset = self.build_inputs(data_)
        logs = model.fit(dataset, epochs=2, steps_per_epoch=3, validation_data=valid_data, validation_steps=1)
        # self.assertIn("loss", logs.history)
        # self.assertIn("accuracy", logs.history)

