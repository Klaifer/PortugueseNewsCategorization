import argparse
import json
import logging
import os.path

from src.common import print_eval, get_dict, getbow, logger
from sklearn.feature_extraction.text import TfidfTransformer
from datetime import datetime
from os.path import basename
from djinn import djinn

def setlog(logfile):
    if logfile:
        logfilename = args.logfile
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
                        default='../data/WikiNews/wikinoticias_train_lemma_stopwords.json',
                        help='Training data')
    parser.add_argument('--test', type=str,
                        default='../data/WikiNews/wikinoticias_test_lemma_stopwords.json',
                        help='Testing data')
    parser.add_argument('--crossvalidation',  type=str,
                        help='Crossvalidation data ids')
    parser.add_argument('--bow',
                        action='store_true',
                        help='use bag-of-words as features. Else, use tf-idf')
    parser.add_argument('--ntrees', type=int, default=50)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--dropout_keep', type=float, default=1.0)
    parser.add_argument('--top_words', type=int, default=2000)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--randomstate', type=int, default=1)
    parser.add_argument('--learnrate', type=float)
    parser.add_argument('--logfile', type=str, help='Output log file')
    parser.add_argument('--modelpath', type=str,
                        default="cpdjinn",
                        help='Output model file')
    parser.add_argument('--load_model', action='store_true', help='Avoid training by reloading checkpoints')


    args = parser.parse_args()
    setlog(args.logfile)
    logger.info(args)

    if not os.path.isdir(args.modelpath):
        os.makedirs(args.modelpath)

    # Carrega os dados
    with open(args.train, 'r') as trainfile, open(args.test, 'r') as testfile:
        train = json.load(trainfile)
        test = json.load(testfile)

    # Preprocessamento
    labels = train["labels"]

    dct = get_dict(train, args.top_words)
    X_train, y_train = getbow(train, dct)
    X_test, y_test = getbow(test, dct)

    if not args.bow:
        # tf-idf
        tfidf = TfidfTransformer().fit(X_train)
        X_train = tfidf.transform(X_train).toarray()
        X_test = tfidf.transform(X_test).toarray()

    if args.crossvalidation:
        with open(args.crossvalidation, 'r') as cvidsfile:
            content = cvidsfile.readlines()
        cvids = [int(l) for l in content]
        X_train = X_train[cvids]
        y_train = y_train[cvids]

    modelname = "class_djinn_test"

    # Classificador
    if not args.load_model:
        logger.info("Train")
        clf = djinn.DJINN_Classifier(args.ntrees, args.max_depth, args.dropout_keep)

        if all([args.batchsize, args.epochs, args.learnrate]):
            optimal = {
                'batch_size':args.batchsize,
                'learn_rate': args.learnrate,
                'epochs': args.epochs
            }
        else:
            logger.info("batchsize, epochs or learnrate undefined. Starting automatic definition. This process can be slow.")
            optimal = clf.get_hyperparameters(X_train, y_train, random_state=args.randomstate)

        batchsize = optimal['batch_size']
        learnrate = optimal['learn_rate']
        epochs = optimal['epochs']

        logger.info("batchsize: {}, learnrate: {}, epochs: {}".format(
            batchsize, learnrate, epochs
        ))

        # train the model with hyperparameters determined above
        clf.train(X_train, y_train, epochs=epochs, learn_rate=learnrate, batch_size=batchsize,
                  display_step=1, save_files=False, file_name=modelname,
                  save_model=True, model_name=modelname, random_state=args.randomstate, model_path=args.modelpath)
    else:
        logger.info("Loading pre-trained")
        clf = djinn.load(model_name=modelname, model_path=args.modelpath)

    logger.info("Eval")
    y_pred = clf.predict(X_test)
    y_true = y_test

    print_eval(y_true, y_pred, labels)
