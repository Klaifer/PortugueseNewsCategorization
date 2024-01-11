import argparse
import json
from src.common import print_eval, get_dict, getbow, logger
from sklearn.feature_extraction.text import TfidfTransformer
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
                        default='../data/WikiNews/wikinews_train_lemma_stopwords.json',
                        help='Training data')
    parser.add_argument('--test', type=str,
                        default='../data/WikiNews/wikinews_test_lemma_stopwords.json',
                        help='Testing data')
    parser.add_argument('--crossvalidation', type=str,
                        default='../data/WikiNews/wikinews_cv0.csv',
                        help='Crossvalidation data ids')
    parser.add_argument('--bow',
                        action='store_true',
                        help='use bag-of-words as features. Else, use tf-idf')
    parser.add_argument('--classifier',
                        choices=['mlp', 'svm'],
                        default='svm')
    parser.add_argument('--logfile', type=str, help='Output log file')

    args = parser.parse_args()
    setlog(args.logfile)
    logger.info(args)

    top_words = 2000

    # Carrega os dados
    with open(args.train, 'r') as trainfile, open(args.test, 'r') as testfile:
        train = json.load(trainfile)
        test = json.load(testfile)

    # Preprocessamento
    labels = train['labels']
    dct = get_dict(train, top_words)
    X_train, y_train = getbow(train, dct)
    X_test, y_test = getbow(test, dct)

    if not args.bow:
        # tf-idf
        tfidf = TfidfTransformer().fit(X_train)
        X_train = tfidf.transform(X_train)
        X_test = tfidf.transform(X_test)

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
