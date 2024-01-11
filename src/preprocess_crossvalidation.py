import json
import logging
import argparse
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S %p',
                    level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str,
                        default='../data/WikiNews/wikinews_train.json',
                        help='Training data')
    parser.add_argument('--nparts', type=int, default=5, help="Number of cross-validation parts.")
    parser.add_argument('--random_seed', type=int, default=0, help="Seed for pseudo random numbers generation.")
    parser.add_argument('--output_prefix', type=str, default="../data/WikiNews/wikinews")

    args = parser.parse_args()
    logging.info(args)

    with open(args.train, 'r') as fin:
        jsoncontent = json.load(fin)
    x = jsoncontent['data']
    y = [d['label'] for d in x]

    skf = StratifiedKFold(n_splits=args.nparts, random_state=args.random_seed, shuffle=True)
    for i, (train_index, _) in enumerate(skf.split(x, y)):
        filename = args.output_prefix+"_cv{}.csv".format(i)
        with open(filename, "w") as partfile:
            for dtid in train_index:
                partfile.write("{}\n".format(dtid))
