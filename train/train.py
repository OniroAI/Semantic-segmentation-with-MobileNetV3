import tensorflow as tf
import numpy as np
import time
import os

from modules.utils import get_model, get_optimizer, get_loss, make_dir


class Model:
    def __init__(self, device, model_name, input_shape, old_model_path=None, **kwargs):
        # Set up model
        self.device = device
        self.model = get_model(model_name, **kwargs)
        init_input = tf.ones(input_shape)
        _ = self.model.predict(init_input)
        if old_model_path is not None:
            self.load(old_model_path)

    def prepare_train(self, train_loader, n_train, val_loader, n_val, loss_name,
                      optimizer, lr, batch_size, max_epoches, save_directory,
                      reduce_factor=None, epoches_limit=0, metrics=None, early_stoping=None,
                      **kwargs):
        self.loss_function = get_loss(loss_name, **kwargs)
        self.lr = lr
        self.optimizer = get_optimizer(optimizer)(learning_rate=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_examples = n_train
        self.val_examples = n_val
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.save_directory = save_directory
        self.reduce_factor = reduce_factor
        self.epoches_limit = epoches_limit
        self.early_stoping = early_stoping
        self.metric_list = None
        if metrics is not None:
            self.metric_list=[]
            for metric in metrics:
                self.metric_list.append(metric)
        make_dir(save_directory)

    def prepare_val(self, val_loader, n_val, loss_name, batch_size,
                      metrics=None, **kwargs):
        self.loss_function = get_loss(loss_name, **kwargs)
        self.val_loader = val_loader
        self.val_examples = n_val
        self.batch_size = batch_size
        self.lr = 0
        self.metric_list = None
        if metrics is not None:
            self.metric_list=[]
            for metric in metrics:
                self.metric_list.append(metric)

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            input, target = batch[0], batch[1]
            prediction = self.model(input, training=True)
            # print(target.shape)
            # print(prediction.shape)
            # print(tf.math.reduce_max(prediction))
            # print(tf.math.reduce_min(prediction))
            loss = self.loss_function(target, prediction)
        # print(self.model.trainable_weights)
        grads = tape.gradient(loss, self.model.trainable_weights)
        # print(grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss.numpy(), prediction, target

    def val_step(self, batch):
        input, target = batch
        prediction = self.model.predict(input)
        loss = self.loss_function(target, prediction)
        if self.metric_list is not None:
            for metric in self.metric_list:
                metric.update(target, prediction)
        return loss.numpy(), prediction, target

    def loader_loop(self, loader, number_of_examples,
                    mode='train'):
        running_loss = 0.0
        if self.metric_list is not None:
            for metric in self.metric_list:
                metric.reset()
        for data in loader:
            if mode == 'train':
                loss_value, pred, target = self.train_step(data)
            elif mode == 'val':
                loss_value, pred, target = self.val_step(data)
            running_loss += loss_value
        running_loss /= number_of_examples# / self.batch_size
        return running_loss

    def fit(self, best_on_val=True):
        # with tf.device(self.device):
        best_loss = float('Inf')
        best_val_loss = float('Inf')
        count_without_improvement = 0
        file_path = ''
        # First validation loop
        c_time = time.time()
        running_val_loss = self.loader_loop(self.val_loader, self.val_examples, mode='val')
        self.print_results(0, 0, running_val_loss, c_time)

        # Train and validation loops
        for epoch in range(self.max_epoches):
            c_time = time.time()
            running_loss = self.loader_loop(self.train_loader, self.train_examples, mode='train')
            running_val_loss = self.loader_loop(self.val_loader, self.val_examples, mode='val')
            self.print_results(epoch, running_loss, running_val_loss, c_time)
            # Saving of the best model
            if best_on_val:
                best_val_loss, count_without_improvement, file_path = self.save_best_model(running_val_loss,
                                                                                           best_val_loss,
                                                                                           count_without_improvement,
                                                                                           file_path)
            else:
                best_loss, count_without_improvement, file_path = self.save_best_model(running_loss,
                                                                                       best_loss,
                                                                                       count_without_improvement,
                                                                                       file_path)
            if self.reduce_factor is not None:
                count_without_improvement = self.reduce_on_plateau(self.reduce_factor,
                                                                   self.epoches_limit,
                                                                   count_without_improvement)
            if self.early_stoping is not None:
                if count_without_improvement > self.early_stoping:
                    print("Stopped. Didn't improve for {} epochs.".format(count_without_improvement))
                    break


        print('Training is finished. Train loss: {}. Best val loss: {}', running_loss, best_val_loss)

    def save_best_model(self, save_loss, save_best_loss, count_without_improvement, old_file_path):
        self.save(self.save_directory + '/model_last.h5')

        if save_loss < save_best_loss:
            count_without_improvement = 0
            print('Model_removed:', old_file_path)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
            new_file_path = self.save_directory + '/model_best_{}.h5'.format(np.round(
                                                                              np.array(save_loss).astype(np.float32),
                                                                              5))
            self.save(new_file_path)
            print('Model_saved:', new_file_path)
            save_best_loss = save_loss
        else:
            count_without_improvement += 1
            new_file_path = old_file_path
        return save_best_loss, count_without_improvement, new_file_path

    def validate(self, val_loader, val_examples):
        c_time = time.time()
        self.model.eval()
        running_val_loss = self.loader_loop(val_loader, val_examples, mode='val')
        self.print_results(0, 0, running_val_loss, c_time)

    def predict(self, input):
        with tf.device(self.device):
            prediction = self.model.predict(input)
            return prediction

    def save(self, path):
        self.model.save_weights(path)
        # pass

    def load(self, path):
        self.model.load_weights(path)

    def print_results(self, epoch, running_loss, running_val_loss, c_time):
        print()
        print(
            'Epoch:', epoch + 1,
            'train_loss:', running_loss,
            'val_loss:', running_val_loss,
            'time:', round(time.time() - c_time, 3),
            's',
            'lr:', self.lr,
        )
        if self.metric_list is not None:
            for metric in self.metric_list:
                value = np.round(float(metric.compute()), 4)
                name = metric.__class__.__name__
                print(name + ': ', value)

    def reduce_on_plateau(self, reduce_factor, epoches_limit, count_without_improvement):
        if count_without_improvement >= epoches_limit:
            count_without_improvement = 0
            self.lr = self.lr*reduce_factor
            self.optimizer.learning_rate = self.lr
        return count_without_improvement






