import json
import logging
import argparse
import string
import spacy
import nltk
import re
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S %p',
                    level=logging.INFO)

nltk.download('rslp')
nlp = spacy.load("pt_core_news_lg")
stemmer = nltk.stem.RSLPStemmer()
ptstopwords = stopwords.words('portuguese')


def prepare(fname, operation, removestopw):
    with open(fname, 'r') as fin:
        jsoncontent = json.load(fin)
    return _prepare_dset(jsoncontent, operation, removestopw)


def _prepare_dset(jsoncontent, operation, removestopw):
    for sample in tqdm(jsoncontent['data']):
        sample['text'] = _prepare_text(sample['text'], operation, removestopw)
    return jsoncontent


def _prepare_text(text, operation, removestopw):
    text = text.lower()
    if operation == "lemma":
        words = _prepare_text_lemma(text, removestopw)
    elif operation == "steam":
        words = _prepare_text_steam(text, removestopw)
    else:
        raise ValueError()

    words = [w for w in words if w not in string.punctuation]
    return " ".join(words)


def _prepare_text_lemma(text, removestopw):
    text = re.sub('\n', ' ', text)

    if removestopw:
        words = word_tokenize(text, language="portuguese")
        words = [w for w in words if w not in ptstopwords]
        text = " ".join(words)

    text = nlp(text)
    words = [t.lemma_ for t in text]

    return words


def _prepare_text_steam(text, removestopw):
    words = word_tokenize(text, language="portuguese")

    if removestopw:
        words = [w for w in words if w not in ptstopwords]

    words = [stemmer.stem(w) for w in words]

    return words


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default="../data/WikiNews/wikinoticias_train.json"
    )

    parser.add_argument(
        '--output',
        type=str,
        default="../data/WikiNews/wikinoticias_train_lemma_stopwords.json"
    )

    parser.add_argument(
        '--operation',
        choices=['steam', 'lemma'],
        default='lemma'
    )

    parser.add_argument(
        '--stopwords',
        action='store_true',
        help='Remove stop words'
    )

    args = parser.parse_args()
    logging.info(args)

    content = prepare(args.input, args.operation, args.stopwords)

    with open(args.output, 'w') as f:
        json.dump(content, f, indent=4)
