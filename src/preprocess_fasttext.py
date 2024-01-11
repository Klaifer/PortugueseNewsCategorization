import json
import logging
import argparse
import fasttext.util
import re
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S %p',
                    level=logging.INFO)

fasttext.util.download_model('pt', if_exists='ignore')  # Portuguese
ft = fasttext.load_model('cc.pt.300.bin')


def prepare(fname):
    with open(fname, 'r') as fin:
        jsoncontent = json.load(fin)
    return _prepare_dset(jsoncontent)


def _prepare_dset(jsoncontent):
    for sample in tqdm(jsoncontent['data']):
        sample['features'] = _prepare_text(sample['text'])
    return jsoncontent


def _prepare_text(text):
    text = re.sub('\n', ' ', text)
    emb = ft.get_sentence_vector(text)
    return emb.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        default="./data/WikiNews/wikinoticias_train_lemma_stopwords.json"
    )

    parser.add_argument(
        '--output',
        type=str,
        default="./data/WikiNews/wikinoticias_train_lemma_stopwords.fasttext.json"
    )

    parser.add_argument(
        '--dimension',
        type=int,
        default=-1,
        help="sentence embeddings size, less than 1 to use default"
    )

    args = parser.parse_args()
    logging.info(args)

    if args.dimension >= 1:
        fasttext.util.reduce_model(ft, args.dimension)

    content = prepare(args.input)
    with open(args.output, 'w') as f:
        json.dump(content, f, indent=4)
