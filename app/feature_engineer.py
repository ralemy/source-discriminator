import os
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime 
from sklearn.preprocessing import LabelEncoder
from app.utils import loggable,INDEV


class FeatureException(Exception):
    '''Holds exceptions during Feature engineering'''
    pass
def throw(s):
    raise FeatureException(s)

class Role(Enum):
    TEST = 1
    TRAIN = 2
    VALIDATE = 3
    PREDICT = 4

@loggable
class FeatureEngineer:
    ''' Reads, cleans, and evaluates the data, prepares a feature set to use in Models or save in Feature Store'''
    def __init__(self, feature_set_name , file_locator):
        ''' initializes the engineer class with the address to the root of data and sets the feature set name and timestamp'''
        self.file_locator = file_locator
        self.features = None 
        self.rows = None
        self.cols = None
        self.timestamp = int(datetime.now().timestamp())
        self.feature_set_name = feature_set_name
        self.silent = not INDEV


    def extract_features(self, training = True):
        self.log('Extracting Features')
        self.read_reference(training)
        if training:
            self.assert_reference()
        self.log('Reading data')
        self.read_data(training)
        self.assert_data(training)
        self.log('Finalizing features')
        self.prepare_feature(training=True)

    def read_reference(self, training = True):
        self.features = pd.read_csv(self.file_locator, header=None)
        self.features.rename(columns = {0:'Path'}, inplace=True)
        self.features['FileId'] = self.features['Path'].apply(lambda x: x[x.rindex('/')+1:x.rindex('.')])
        if not training: 
            return
        self.features.rename(columns = {1:'Label'}, inplace=True)        
        self.features[['Tag', 'Source']] = self.features.apply(lambda x: x['FileId'].split('_')[:2], axis = 1, result_type='expand')
        self.features['Subject'] = LabelEncoder().fit_transform(self.features['Source']) 

    def assert_reference(self):
        df = self.features
        tags = df['Tag'].value_counts()
        labels = df['Label'].value_counts()
        assert(df[(df['Tag']=='MW')^(df['Label']==1)].shape[0] == 0 or throw('mismatch tag and value'))
        assert(len(tags.values)== 2 or throw('too many tags'))
        assert('F' in tags.index.values or throw('no F tag'))
        assert('MW' in tags.index.values or throw('no MW tag'))
        assert(len(labels.values)==2 or throw('too many labels'))
        assert(0 in labels.index.values or throw('no 0 label'))
        assert(1 in labels.index.values or throw('no 1 lable'))

    def read_data(self,training=True):
        def read_matrix(df):
            with open(df['Path'],"r") as f:
                inline = '' if not training else f.readline().rstrip()[2:]
                matrix = np.array([[float(v) for v in line.strip().split(',')] for line in f.readlines()])
            return inline,matrix,matrix.shape[0],matrix.shape[1]
        self.features[['InlineTag','Matrix','Rows','Columns']] = self.features.apply(read_matrix, axis=1, result_type='expand')
        self.rows = self.features.loc[0]['Rows']
        self.cols = self.features.loc[0]['Columns']

    def assert_data(self, training=True):
        df = self.features
        if training:
            assert(df[df.Tag != df.InlineTag].shape[0] == 0 or throw('mismatch tag in file name and content '))
        assert(df.Rows.nunique() == 1 or throw('different matrix rows count'))
        assert(df.Columns.nunique() == 1 or throw('different matrix columns count'))

    def prepare_feature(self, training = True):
        self.features['timestamp'] = self.timestamp
        self.features.drop(['Path', 'InlineTag', 'Rows', 'Columns'], axis=1, inplace=True)
        if training:
            self.features.drop(['Tag', 'Source'], axis=1, inplace=True)
        
        self.features['Matrix'] = self.features['Matrix'].apply(lambda x: ((x-np.min(x))/np.ptp(x)).tolist())

    def split(self, train, test=None, val=None):
        probs = np.random.rand(len(self.features))
        if test is not None and val is not None:
            test_mask = (probs>=train) & (probs < (train + test))
        else:
            test_mask = probs >= train
        if val is not None:
            validation_mask = probs >= 1-val
        else:
            validation_mask = probs < 0 # i.e. no validation

        self.features['role'] = Role.TRAIN
        self.features.loc[test_mask, 'role'] = Role.TEST
        self.features.loc[validation_mask, 'role'] = Role.VALIDATE

    def get_data_set(self, role=Role.TRAIN):
        if 'role' not in self.features:
            return self.features
        return self.features[self.features.role == role]

    def get_matrix(self, df):
        return df['Matrix']

    def get_label(self, df):
        return df['Label']

    def get_subject(self,df):
        return df['Subject']

    def save_feature_set(self, store_location):
        destination = os.path.join(store_location, self.feature_set_name, str(self.timestamp))
        Path(destination).mkdir(parents=True, exist_ok=True)
        self.log('Pickling the feature set')
        self.features.to_pickle(os.path.join(destination, 'features.pickle'), compression=None)
        self.log('Pickling the config')
        pd.to_pickle({'timestamp': self.timestamp,
                        'rows': self.rows, 
                        'cols': self.cols, 
                        'set_name': self.feature_set_name 
                    }, os.path.join(destination, 'config.pickle'))

    def load_feature_set(self, destination):
        self.log('reading the feature set')
        self.features = pd.read_pickle(os.path.join(destination, 'features.pickle'), compression=None)
        self.log('reading the config')
        config = pd.read_pickle(os.path.join(destination, 'config.pickle'))
        self.timestamp = config['timestamp']
        self.rows = config['rows']
        self.cols = config['cols']
        self.feature_set_name = config['set_name']

    def get_id(self):
        return self.timestamp




