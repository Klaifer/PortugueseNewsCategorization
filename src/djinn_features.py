import argparse
import json
import numpy as np
from src.common import print_eval, logger
import logging
from datetime import datetime
from os.path import basename
from djinn import djinn
import os

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
    parser.add_argument('--ntrees', type=int, default=50)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--dropout_keep', type=float, default=1.0)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--randomstate', type=int, default=1)
    parser.add_argument('--learnrate', type=float)
    parser.add_argument('--logfile', type=str, help='Output log file')
    parser.add_argument('--modelpath', type=str, help='Output log file', default="./")
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
    labels = train['labels']
    x_train = np.asarray([d['features'] for d in train['data']])
    y_train = np.asarray([d['label'] for d in train['data']])
    x_test = np.asarray([d['features'] for d in test['data']])
    y_test = np.asarray([d['label'] for d in test['data']])

    if args.crossvalidation:
        with open(args.crossvalidation, 'r') as cvidsfile:
            content = cvidsfile.readlines()
        cvids = [int(l) for l in content]
        x_train = x_train[cvids]
        y_train = y_train[cvids]


    modelname = "class_djinn_test"  # name the model

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
            logger.info(
                "batchsize, epochs or learnrate undefined. Starting automatic definition. This process can be slow.")
            optimal = clf.get_hyperparameters(x_train, y_train, random_state=args.randomstate)

        batchsize = optimal['batch_size']
        learnrate = optimal['learn_rate']
        epochs = optimal['epochs']

        logger.info("batchsize: {}, learnrate: {}, epochs: {}".format(
            batchsize, learnrate, epochs
        ))

        # train the model with hyperparameters determined above
        clf.train(x_train, y_train, epochs=epochs, learn_rate=learnrate, batch_size=batchsize,
                  display_step=1, save_files=True, file_name=modelname,
                  save_model=True, model_name=modelname, random_state=args.randomstate, model_path=args.modelpath)
    else:
        logger.info("Loading pre-trained")
        clf = djinn.load(model_name=modelname, model_path=args.modelpath)

    logger.info("Eval")
    y_pred = clf.predict(x_test)
    y_true = y_test

    print_eval(y_true, y_pred, labels)
