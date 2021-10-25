import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import optimizers, metrics, losses
from tensorflow.python.keras.backend import update

from app.utils import loggable, generate_session_id
from app.feature_engineer import FeatureEngineer, Role
from app.components import Encoder, Predictor, Discriminator
from app.tensor_metrics import TensorMetrics

@loggable
class ZhaoModel:
    ''' Trains the architecture based on the cooperative game between encoder-predictor and minmax game between discriminator the the other two
        Predicts labels from a trained checkpoint given new data.

        initialized with a dictionary: 
            loss_lambda: coefficient to apply to discrimator loss to find the total loss
            data_path: where to find the data. if refresh data is False, points to feature store endpoint. else to root directory of raw data
            store_path: where to find the feature store. used to save features after they are engineered
            log_path: where to log tensorboard data
            checkpoint_path: where to save the model checkpoints
            pred_path: where to save the predictions
            epochs, 
            batch_size, 
            refresh_data if False, data_path points to data already processed. else, data should be processed and stored in feature store
            set_name the feature set name for this model
    '''
    def __init__(self, options, training=True):
        self.loss_lambda = 1.2 if 'loss_lambda' not in options else float(options['loss_lambda'])
        self.data_path = '../feature_store/someset/1634831542' if 'data_path' not in options else options['data_path']
        self.store_path = '../feature_store' if 'store_path' not in options else options['store_path']
        self.log_path = os.path.join('..','logs', 'gradient_tape') if 'log_path' not in options else options['log_path']
        self.checkpoint_path = '../checkpoint' if 'checkpoint_path' not in options else options['checkpoint_path']
        self.pred_path = '../predictions' if 'pred_path' not in options else options['pred_path']
        self.epochs = 5 if 'epochs' not in options else int(options['epochs'])
        self.learning_rate = 0.001 if 'learning_rate' not in options else float(options['learning_rate'])
        self.batch_size = 32 if 'batch_size' not in options else int(options['batch_size'])
        self.refresh_data = False if 'refresh_data' not in options else options ['refresh_data'].upper() == 'TRUE'
        self.set_name = 'source_discrimator' if 'set_name' not in options else options['set_name']
        self.feature_set = self.get_feature_set(training)
        self.encoder = Encoder()
        self.discriminator = Discriminator()
        self.predictor = Predictor()
        self.loss_obj = losses.BinaryCrossentropy(from_logits= True, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE) 
        self.optimizer = optimizers.Adam(learning_rate=self.learning_rate_fn())
        self.checkpoint=tf.train.Checkpoint(encoder=self.encoder, predictor=self.predictor)
        self.h_subject = None # H(s) required by inner loop. will be calculated in training
        self.tmetrics = TensorMetrics(metrics.Mean, metrics.BinaryAccuracy, self.log_path)

    def predict(self):
        df = self.feature_set.get_data_set(Role.PREDICT)
        self.log('restoring checkpoint', self.checkpoint_path)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_path))
        self.log('predicting labels')
        predictions= self.test_step(self.feature_set.get_matrix(df), None, False)
        df['Predictions'] = predictions
        output = os.path.join(self.pred_path, generate_session_id(), 'model_predictions.pickle')
        self.log('saving predictions in', output)
        df.drop('Matrix').to_csv(output, sep=',')
        return predictions

    def train(self, train_frac, test_frac, validation_frac):
        fe = self.feature_set
        fe.split(train_frac, test_frac, validation_frac)
        train_df = fe.get_data_set(Role.TRAIN)
        val_df = fe.get_data_set(Role.VALIDATE)

        train_size = train_df.shape[0]
        self.log('reading training set')
        train_set = self.get_data_set(train_df, self.batch_size, True)
        val_set = self.get_data_set(val_df, self.batch_size, False)
        self.h_subject = self.get_entropy(train_df, 'Subject')
        self.log('running epochs')
        self.run_epochs(train_set, val_set, train_size // self.batch_size, val_df.shape[0]//self.batch_size)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_path))
        self.log('gathering final results')
        return self.final_results()

    def get_feature_set(self, training=True):
        fe = FeatureEngineer(self.set_name, self.data_path)
        if self.refresh_data:
            self.log('Refresh Data', self.set_name, self.data_path)
            fe.extract_features(training)
            self.log('Saving Data', self.store_path, fe.feature_set_name, str(fe.timestamp))
            fe.save_feature_set(self.store_path)
        else:
            self.log('Loading from feature store', self.data_path)
            fe.load_feature_set(self.data_path)
        return fe

    def reshape_for_cnn(self, df):
        fe = self.feature_set
        return np.array(fe.get_matrix(df).values.tolist()).reshape(df.shape[0], fe.rows, fe.cols, 1)

    def get_data_set(self, df, batch_size, shuffle=False):
        size = df.shape[0]
        fe = self.feature_set
        data_set = tf.data.Dataset.from_tensor_slices((
            self.reshape_for_cnn(df),
            fe.get_label(df).values,
            fe.get_subject(df).values
        ))
        if shuffle:
            data_set = data_set.shuffle(size, reshuffle_each_iteration=True)
        return data_set.batch(batch_size)

    def learning_rate_fn(self):
        return optimizers.schedules.ExponentialDecay(self.learning_rate, 100000, 0.96, True, 'exp_decay_lr' )
                
    def get_entropy(self,df, col):
        '''calculate entropy for distinct variable'''
        v = df[col].value_counts()
        s= v.sum()
        return v.apply(lambda x: x/s).apply(lambda x: -x * np.log2(x)).sum()

    def final_results(self):
        fe = self.feature_set
        loss_metric = self.tmetrics.metrics['loss']['test']
        acc_metric = self.tmetrics.metrics['accuracy']['test']
        metrics = {}
        for key in [Role.VALIDATE, Role.TRAIN, Role.TEST]:
            self.update_final_metric(metrics, fe.get_data_set(key), key.name)
        
        metrics['Encodings'] = {
            'output': self.final_result(fe.get_data_set(Role.TRAIN), loss_metric, acc_metric, True),
            'labels': fe.get_label(fe.get_data_set(Role.TRAIN)).values
        }
        return metrics

    def update_final_metric(self, metrics, df,key):
        loss_metric = self.tmetrics.metrics['loss']['test']
        acc_metric = self.tmetrics.metrics['accuracy']['test']
        self.final_result(df, loss_metric, acc_metric)
        metrics[key] = {'loss': loss_metric.result(), 'accuracy': acc_metric.result()}


    def final_result(self,df, loss, acc, report = False):
        loss.reset_states()
        acc.reset_states()
        steps = df.shape[0] // self.batch_size
        data_set = self.get_data_set(df, self.batch_size, False)
        feature_set = None
        for data,label,_ in data_set.take(steps):
            features = self.test_step(data, label).numpy()
            if report:
                feature_set = features if feature_set is None else np.vstack((feature_set, features))
        return feature_set

    # Here starts the training loop. the run_epochs function 
    # Is not responsible for algorithm implementation details
    # It just runs each epoch, checks its accuracy and saves the 
    # best one.

    def run_epochs(self, train_set, val_set, train_step, val_step):
        max_acc = -100.0
        for epoch in range(self.epochs):
            self.log('epoch', epoch, 'from', self.epochs)
            self.tmetrics.reset_epoch_metrics()
            for index, (data, labels, subjects) in enumerate(train_set.take(train_step)):
                self.run_step(epoch, data, labels, subjects, index)
            self.log('done. validating....')
            self.debug('test_acc_before_val', self.tmetrics.metrics['accuracy']['test'].result())
            for vd, vl, _ in val_set.take(val_step):
                self.test_step(vd, vl)
            epoch_acc = self.tmetrics.report_epoch(epoch)
            self.debug('test_acc_after_val', self.tmetrics.metrics['accuracy']['test'].result())
            if max_acc < epoch_acc:
                self.checkpoint.save(file_prefix=os.path.join(self.checkpoint_path, 'check_point.ckpt'))
                max_acc = epoch_acc
                self.log('Best Accuracy so far', max_acc)
            else:
                self.log('accuracy', epoch_acc, 'best_accuracy', max_acc)
    
    # Here starts the implementation of the alogorithm
    #  Matched to Algorithm on page 6 reference article
    def run_step(self, epoch, data, labels, subjects, index):
        ''' update encoder, predictor, and discriminator'''
        with tf.GradientTape(persistent=True) as tape:
            e_x = self.encoder(data, training=True)
            w_i = self.predictor(e_x, training=False)
            l_p = self.loss_obj(labels, w_i) 
            di = tf.concat([w_i, e_x], 1)
            q_d = self.discriminator(di, training=True)
            l_d = self.loss_obj(subjects, q_d)

        # with tf.GradientTape() as d_tape:
        #     l_d, q_d = self.get_disc_loss(subjects, e_x, w_i)
        #     v_i  = l_p - (self.loss_lambda * l_d)
            v_i = l_p
        

        self.update_model(self.encoder, tape, v_i)
        self.update_model(self.predictor, tape, v_i)
        self.update_model(self.discriminator, tape, v_i, 'max')
        # round=0
        # while True:
        #     self.update_model(self.discriminator, d_tape, v_i, 'max')
        #     with tf.GradientTape() as disc_tape:
        #         l_d, q_d = self.get_disc_loss(subjects, e_x, w_i)
        #         v_i = l_p - self.loss_lambda * l_d
        #     if l_d <= self.h_subject:
        #         break
        #     d_tape=disc_tape
        #     round+=1
        if index % 20 == 0:
            self.log('batch', index)        

        # self.tmetrics.update_loss('global', v_i)
        # self.tmetrics.update_accuracy('global', subjects, q_d)
        self.tmetrics.update_loss('train', l_p)
        self.tmetrics.update_accuracy('train', labels, w_i)

    def test_step(self, values, expected, training=True):
        enc_actual = self.encoder(values, training=False)
        predictions = self.predictor(enc_actual, training=False)
        if not training:
            return predictions
        loss = self.loss_obj(expected, predictions)
        self.tmetrics.update_loss('test', loss)
        self.tmetrics.update_accuracy('test', expected, predictions)
        self.debug('test_acc_in_step', self.tmetrics.metrics['accuracy']['test'].result(), loss)
        return enc_actual
        
    def get_disc_loss(self, subjects, enc_output, pred_output):
        '''get the loss of discriminator afte conditioning with predictor output'''
        disc_input = tf.concat([pred_output, enc_output], 1)
        q_d = self.discriminator(disc_input, training=True)
        return self.loss_obj(subjects, q_d), q_d


    def update_model(self,model, tape, loss, minmax='min'):
        '''Update weights based on loss and tape gradient'''
        vars = model.trainable_variables
        if minmax == 'max':
            grads = [-x for x in tape.gradient(loss, vars)]
        else:
            grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads,vars))
    

