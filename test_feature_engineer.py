import argparse
from app.feature_engineer import FeatureEngineer
import os
from datetime import date, datetime

parser = argparse.ArgumentParser(description='test loading and saving feature store')

# example load: ../feature_store/someset/1634831542 

parser.add_argument('--action', 
                    help='load or save')
parser.add_argument('--place', 
                    help='where to save')

args = parser.parse_args()

if args.action == 'save':
    print('creating data ', datetime.now())
    fe = FeatureEngineer('someset', '../data/file_locator.csv')
    fe.extract_features()
    print(fe.features.head())
    print('saving to', os.path.join(args.place, fe.feature_set_name, str(fe.timestamp)), datetime.now())
    fe.save_feature_set(args.place)
    print('done', datetime.now())

elif args.action == 'load':
    print('loading data', datetime.now())
    fe = FeatureEngineer('someset', '../data/file_locator.csv')
    fe.load_feature_set(args.place)
    print(fe.features.head(), datetime.now())
    print(fe.feature_set_name, fe.timestamp, fe.data_dim)
    print('done')