import argparse
import itertools
import json
import os
import logging
import torch
from datetime import datetime
from os.path import basename
from sklearn.model_selection import StratifiedKFold
from cnnclassifier import load_data, pad_sequence, load_pretrainned_embeddings, MyCNN, train, logresults, logger

PAD = 0  # Precisa ser 0, porque é o valor para preencher a janela da convolução
UNK = 1


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

def get_experiments(atb_range):
    labels = []
    values = []
    for k in atb_range:
        labels.append(k)
        values.append(atb_range[k])
    combinations = [v for v in itertools.product(*values)]
    pvalues = [{k: v for k, v in zip(labels, curr)} for curr in combinations]

    return pvalues


def optimize(paramset, trainfile, vocabulary_size=2000, sentence_length=200, nparts=5):
    try:
        os.makedirs("cpcnnopt")
    except FileExistsError:
        pass

    experiments = get_experiments(paramset)

    x, y, y_labels, labels, word2idx = load_data(trainfile, vbsize=vocabulary_size)
    x = pad_sequence(x, maxlen=sentence_length, pad_char=PAD, pad_post=False)
    embeddings_arr = load_pretrainned_embeddings(word2idx)
    skf = StratifiedKFold(n_splits=nparts, random_state=0, shuffle=True)

    for istep, (train_index, valid_index) in enumerate(skf.split(x, y_labels)):
        logger.info("Step {}".format(str(istep)))

        x_train = x[train_index]
        y_train = y[train_index]

        x_valid = x[valid_index]
        y_valid = y[valid_index]

        for params in experiments:
            logger.info("Params {}".format(json.dumps(params)))

            if params['emb'] == 'fasttext':
                preembeddings = embeddings_arr
                embedding_size = None  # unused, can be any value
            else:
                preembeddings = None
                embedding_size = params['emb']

            cnn = MyCNN(nlbs=len(labels), emb_arr=preembeddings, sentlen=sentence_length, nwords=vocabulary_size,
                        freeze_emb=False, emb_dim=embedding_size, nconv=params['conv'], dropout=params['dout'])

            checkpointfile = "cpcnnopt/checkpoint_wiki_{}.pt".format(istep)
            train(cnn, x_train, y_train, x_valid, y_valid, checkpointfile, epochs=25, patience=4)

            # Print evaluations with validation data
            cnn.load_state_dict(torch.load(checkpointfile))
            logresults(cnn, x_valid, y_valid, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default='../data/WikiNews/wikinews_train_lemma_stopwords.json',
                        help='Training data')
    parser.add_argument("--vocabulary_size", type=int, default=2000)
    parser.add_argument("--sentence_length", type=int, default=200)
    parser.add_argument('--logfile', type=str, help='Output log file')

    args = parser.parse_args()
    setlog(args.logfile)
    logger.info(args)

    params = {
        'emb': [32, 64, 128, 256, 'fasttext'],
        'conv': [64, 128, 256],
        'dout': [0.1, 0.2, 0.3]
    }

    optimize(params, args.train, args.vocabulary_size, args.sentence_length)
