#!/usr/bin/env python
# ------------------------------------------------------
# Confusing the effect of source using Conditional Adversarial Architecture
# On a binary classification system
# Reference: Zhao M, et.al ICML 2017
# Adapted by: Reza Alemy 
# email: reza@alemy.net

import argparse
import configparser

import pickle
from app.zhao_model import ZhaoModel
from app.plot import Plotter
from app.utils import generate_session_id
import os

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', choices=['train', 'predict'], required=True)
    parser.add_argument('-t', '--config', action='store', help='Path to config file for trainig or prediction', required=True)
    return parser

def read_config(file_path, action):
    config = configparser.ConfigParser()
    config.read(file_path)
    return config[action]

def report_metrics(metrics, options):
    for k, v in metrics.items():
        if k != 'Encodings':
            print('Metrics for ', k, v)
    plotter = Plotter(options)
    plotter.plot_components(metrics['Encodings']['output'], metrics['Encodings']['labels'])
    
def save_info(options, metrics, tensor_log):
    base = os.path.join(options['history_path'], options['sessionId'])
    metrics['tensorboard'] = tensor_log
    with open(os.path.join(base, 'options.pkl'), 'wb') as f:
        pickle.dump(options, f)
    with open(os.path.join(base, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    pass

if __name__=='__main__':
    parser = init_args()
    args = parser.parse_args()
    config = read_config(args.config, args.action.upper())
    options = {k:config[k] for k in config}
    options['sessionId']= generate_session_id()
    if args.action.upper() == 'TRAIN':
        model = ZhaoModel(options)
        model.error('Starting to Train. Tensorboard can be found in:', model.tmetrics.log_base)
        metrics = model.train(0.8, 0.1, 0.1)
        report_metrics(metrics, options)
        save_info(options, metrics, model.tmetrics.log_base)
        model.log('training complete')
    elif args.action.upper() == 'PREDICT':
        model = ZhaoModel(options)
        model.predict()




