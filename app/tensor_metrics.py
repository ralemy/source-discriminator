from datetime import datetime
import os
import tensorflow as tf

class TensorMetrics:
    def __init__(self, loss_metric, accuracy_metric, log_path, sessionId) -> None:
        self.metrics = {
            'loss': {
                'test': loss_metric(name = 'test_loss'),
                'train': loss_metric(name = 'train_loss'),
                'global': loss_metric(name = 'global_loss')
            },
            'accuracy': {
                'test': accuracy_metric(name = 'test_acc'),
                'train': accuracy_metric(name = 'train_acc'),
                'global': accuracy_metric(name = 'global_acc')
            }
        }
        self.log_base = os.path.join(log_path, sessionId)
        os.mkdir(self.log_base)
        self.summary_writer = {
            'train': tf.summary.create_file_writer(os.path.join(self.log_base,  'train')),
            'test': tf.summary.create_file_writer(os.path.join(self.log_base,  'test')),
            'global': tf.summary.create_file_writer(os.path.join(self.log_base, 'global'))
        }

    def reset_epoch_metrics(self):
        for x in ['train', 'test']:
            for y in ['loss', 'accuracy']:
                self.metrics[y][x].reset_states()

    def write_metric_set(self, epoch, role='train'):
        with self.summary_writer[role].as_default():
            tf.summary.scalar('loss', self.metrics['loss'][role].result(), step=epoch)
            tf.summary.scalar('accuracy', self.metrics['accuracy'][role].result(), step=epoch)

    def update_loss(self, role, loss):
        self.metrics['loss'][role](loss)

    def update_accuracy(self, role, expected, actual):
        self.metrics['accuracy'][role](expected, actual)

    def report_epoch(self,epoch):
        self.write_metric_set(epoch, 'train')
        self.write_metric_set(epoch, 'test')
        self.write_metric_set(epoch, 'global')
        return self.metrics['accuracy']['test'].result()
