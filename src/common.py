from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
import numpy as np
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S %p',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


def print_eval(y_true, y_pred, labels):
    result = {}
    result['micro'] = f1_score(y_true, y_pred, average='micro')
    result['macro'] = f1_score(y_true, y_pred, average='macro')
    result['weighted'] = f1_score(y_true, y_pred, average='weighted')


    logger.info("micro")
    logger.info(result['micro'])
    logger.info("macro")
    logger.info(result['macro'])
    logger.info("weighted")
    logger.info(result['weighted'])

    cmatrix = confusion_matrix(y_true, y_pred, labels=[i for i in range(len(labels))])
    logger.info(pd.DataFrame(cmatrix, index=labels, columns=[l[:5] for l in labels]))
    return result


def get_dict(content, nwords=2000):
    dct = Counter()
    for d in content['data']:
        dct.update(d['text'].split())

    dct = {k: i for i, (k, _) in enumerate(dct.most_common(nwords))}

    return dct


def getbow(jsoncontent, dct):
    x = []
    y = []
    nwords = len(dct)
    for d in jsoncontent['data']:
        sample = np.zeros(nwords)
        text = d['text'].split()
        for w in text:
            if w in dct:
                sample[dct[w]] += 1

        x.append(sample)
        y.append(d['label'])

    return np.array(x), np.array(y)
