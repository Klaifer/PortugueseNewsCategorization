import argparse
import json
import numpy as np
from src.common import print_eval, logger
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import logging
from datetime import datetime
from os.path import basename


def setlog(logfile):
    if logfile:
        logfilename = logfile
    else:
        scriptname = basename(__file__).split(".")[0]
        logfilename = '../logs/{}_{:%Y-%m-%d_%H-%M-%S}.log'.format(scriptname, datetime.now())

    logfile = logging.FileHandler(logfilename)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S %p'
    )
    logfile.setFormatter(formatter)
    logfile.setLevel(logging.INFO)
    logger.addHandler(logfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default='../data/WikiNews/wikinoticias_train_lemma_stopwords.fastext.json',
                        help='Training data')
    parser.add_argument('--test', type=str,
                        default='../data/WikiNews/wikinoticias_test_lemma_stopwords.fastext.json',
                        help='Testing data')
    parser.add_argument('--crossvalidation',  type=str,
                        help='Crossvalidation data ids')
    parser.add_argument('--classifier',
                        choices=['mlp', 'svm'],
                        default='svm')
    parser.add_argument('--logfile', type=str, help='Output log file')

    args = parser.parse_args()
    setlog(args.logfile)
    logger.info(args)

    # Carrega os dados
    with open(args.train, 'r') as trainfile, open(args.test, 'r') as testfile:
        train = json.load(trainfile)
        test = json.load(testfile)

    # Preprocessamento
    labels = train['labels']
    X_train = np.array([d['features'] for d in train['data']])
    y_train = np.array([d['label'] for d in train['data']])
    X_test = np.array([d['features'] for d in test['data']])
    y_test = np.array([d['label'] for d in test['data']])

    if args.crossvalidation:
        with open(args.crossvalidation, 'r') as cvidsfile:
            content = cvidsfile.readlines()
        cvids = [int(l) for l in content]
        X_train = X_train[cvids]
        y_train = y_train[cvids]

    # Classificador
    logger.info("Train")
    if args.classifier == 'svm':
        clf = SVC(kernel='rbf', decision_function_shape='ovr')
    else:
        clf = MLPClassifier()
    clf.fit(X_train, y_train)

    logger.info("Eval")
    y_pred = clf.predict(X_test)
    y_true = y_test

    print_eval(y_true, y_pred, labels)
