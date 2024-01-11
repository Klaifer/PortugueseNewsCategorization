import logging
import torch
from sklearn.metrics import f1_score
import argparse
import json
from collections import Counter
import numpy as np
from src.common import print_eval, logger
import fasttext.util
from datetime import datetime
from os.path import basename
from torch.utils.data import TensorDataset, RandomSampler, DataLoader


PAD = 0  # Precisa ser 0, porque é o valor para preencher a janela da convolução
UNK = 1

def get_word2idx(content, nwords=800):
    w2i = Counter()
    for d in content['data']:
        w2i.update(d['text'].split())

    w2i = {k: i for i, (k, c) in enumerate(w2i.most_common(nwords), 2)}
    w2i['<pad>'] = PAD
    w2i['<unk>'] = UNK

    return w2i


def sentences2idxs(jsoncontent, w2i, nclass, remove_missing=True):
    """
    Change words to tokens ids

    :param jsoncontent:
    :param w2i:
    :param nclass:
    :param remove_missing: True to remove, False to change to unk
    :return:
    """
    x = []
    y = []
    for d in jsoncontent['data']:
        text = d['text'].split()
        if remove_missing:
            textid = [w2i[w] for w in text if w in w2i]
        else:
            textid = [w2i[w] if w in w2i else UNK for w in text]

        label = np.zeros(nclass)
        label[d['label']] = 1

        x.append(textid)
        y.append(label)

    return x, np.array(y)


def load_pretrainned_embeddings(w2i):
    ft = fasttext.load_model('cc.pt.300.bin')
    demb = ft.get_dimension()

    dctemb = np.zeros((len(w2i), demb))
    # dctemb[PAD] = np.zeros((demb,))
    dctemb[UNK] = np.ones((demb,))

    for word, key in w2i.items():
        if key in [PAD, UNK]:
            continue

        dctemb[key] = ft.get_word_vector(word)

    return dctemb


class MyCNN(torch.nn.Module):
    def __init__(self, nlbs, emb_arr=None, emb_dim=64, dropout=0.1, sentlen=500, nwords=2000, nconv=128,
                 freeze_emb=True):
        super(MyCNN, self).__init__()
        self.sentlen = sentlen
        self.nwords = nwords

        # ----- In ----
        if emb_arr is None:
            self.emb = torch.nn.Embedding(nwords + 2, emb_dim)  # topwords + pad + unk
        else:
            emb_dim = emb_arr.shape[1]
            self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(emb_arr), freeze=freeze_emb)

        # ----- conv ----
        # 'padding' é o tamanho do padding da janela. 'same' é o tamanho necessário para manter a entrada do mesmo
        # tamanho da saída. O valor que será utilizado com padding é definido em 'padding_mode'. O default é 'zeros',
        # que é o mesmo valor que estou usando para padding nas sentenças
        self.conv2 = torch.nn.Conv1d(emb_dim, nconv, 2, padding='valid')
        self.conv3 = torch.nn.Conv1d(emb_dim, nconv, 3, padding='valid')
        self.conv4 = torch.nn.Conv1d(emb_dim, nconv, 4, padding='valid')
        self.conv5 = torch.nn.Conv1d(emb_dim, nconv, 5, padding='valid')

        # Full connected layer
        # As saídas dos filtros receberam max operation, resultando em uma saída por filtro
        self.fc = torch.nn.Linear(nconv * 4, nlbs)

        # Dropout
        self.dropout = torch.nn.Dropout(dropout)

        # Output
        # o parâmetro dim é a dimensão da matrix que será aplicado. O default é a camada mais profunda
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.emb(x).float()
        x = x.transpose(1, 2).contiguous()

        x2 = torch.nn.functional.relu(self.conv2(x))
        x3 = torch.nn.functional.relu(self.conv3(x))
        x4 = torch.nn.functional.relu(self.conv4(x))
        x5 = torch.nn.functional.relu(self.conv5(x))

        x2 = torch.nn.functional.max_pool1d(x2, kernel_size=x2.shape[2]).squeeze(dim=2)
        x3 = torch.nn.functional.max_pool1d(x3, kernel_size=x3.shape[2]).squeeze(dim=2)
        x4 = torch.nn.functional.max_pool1d(x4, kernel_size=x4.shape[2]).squeeze(dim=2)
        x5 = torch.nn.functional.max_pool1d(x5, kernel_size=x5.shape[2]).squeeze(dim=2)

        x = torch.cat([x2, x3, x4, x5], dim=1)
        x = self.dropout(x)

        x = self.fc(x)  # Multiplicação com pesos sinapticos ajustáveis e bias
        x = self.softmax(x)  # Converte as ativações dos neurônios em aproximações da probabilidade

        return x


def get_comparison_arrays(model, dataloader):
    model.eval()
    y_expected = None
    y_predicted = None
    total_loss = 0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batches += 1
            x_batch, y_batch = tuple(t for t in batch)
            y_batch = y_batch.type(torch.FloatTensor)
            y_pred = model(x_batch)
            total_loss += torch.nn.functional.binary_cross_entropy(y_pred, y_batch).numpy()

            y_batch = y_batch.numpy()
            y_expected = y_batch if y_expected is None else np.append(y_expected, y_batch, axis=0)

            y_argmax = torch.argmax(y_pred, dim=1)
            y_pred = torch.zeros_like(y_pred).scatter_(1, y_argmax.unsqueeze(1), 1.)
            y_pred = y_pred.detach().numpy()

            y_predicted = y_pred if y_predicted is None else np.append(y_predicted, y_pred, axis=0)

    return y_expected, y_predicted, total_loss / batches


def evaluate(model, dataloader):
    y_expected, y_predicted, mean_loss = get_comparison_arrays(model, dataloader)
    return f1_score(y_expected, y_predicted, average='micro'), mean_loss


def logresults(model, x, y, labels):
    data = get_dataloader(x, y)
    y_expected, y_predicted, mean_loss = get_comparison_arrays(model, data)
    y_expected = np.argmax(y_expected, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)

    print_eval(y_expected, y_predicted, labels)
    logger.info("End")


def get_dataloader(x, y, bsize=128):
    train_data = TensorDataset(torch.tensor(x), torch.tensor(y))
    train_sampler = RandomSampler(train_data)
    return DataLoader(train_data, sampler=train_sampler, batch_size=bsize)


def train(model, train_data, train_labels, monitor_data, monitor_labels, checkpoint="checkpoint", epochs=10,
          patience=3):
    train_dataloader = get_dataloader(train_data, train_labels)
    monitor_dataloader = get_dataloader(monitor_data, monitor_labels)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Tracking best validation accuracy
    best_loss = np.inf
    patience_count = 0

    for epoch_i in range(epochs):
        # Tracking time and loss
        step_loss = 0
        step = 0
        patience_count += 1

        model.train()
        for step, batch in enumerate(train_dataloader):
            x_batch, y_batch = tuple(t for t in batch)
            y_batch = y_batch.type(torch.FloatTensor)

            # Clean gradientes
            optimizer.zero_grad()

            # Feed the model
            y_pred = model(x_batch)

            # Loss calculation
            loss = torch.nn.functional.binary_cross_entropy(y_pred, y_batch)
            step_loss += loss.item()

            # Gradients calculation
            loss.backward()
            del loss
            del y_pred

            # Gradients update
            optimizer.step()



        step_loss /= (step+1)

        # Evaluation phase
        if monitor_data is not None:
            f1, test_loss = evaluate(model, monitor_dataloader)
            logger.info("Epoch: %d, train_loss: %.5f, valid_loss: %.5f, valid_mean_f1: %.5f" % (
                epoch_i + 1, step_loss, test_loss, f1))
            step_loss = test_loss

        if best_loss > step_loss:
            logger.info("Saving best model")
            best_loss = step_loss
            torch.save(model.state_dict(), checkpoint)
            patience_count = 0

        if patience_count == patience:
            logging.info("Early stop on step " + str(epoch_i))
            break


def load_data(filename, w2i=None, vbsize=2000):
    with open(filename, 'r') as datafile:
        jsoncontent = json.load(datafile)

    lb_names = jsoncontent['labels']
    if not w2i:
        w2i = get_word2idx(jsoncontent, vbsize)

    x, y_multilabel = sentences2idxs(jsoncontent, w2i, len(lb_names))
    y_label = [d['label'] for d in jsoncontent['data']]

    return x, y_multilabel, y_label, lb_names, w2i


def pad_sequence(sequences, maxlen, pad_char=0, pad_post=False, truncate_post=False, dtype="int32"):
    """
    Some code borrowed from "tensorflow"

    :param sequences:
    :param maxlen:
    :param pad_char:
    :param pad_post:
    :param truncate_post:
    :param dtype:
    :return:
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.unicode_
    )
    if isinstance(pad_char, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(pad_char)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen), pad_char, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncate_post:
            trunc = s[:maxlen]
        else:
            trunc = s[-maxlen:]

        if pad_post:
            x[idx, : len(trunc)] = trunc
        else:
            x[idx, -len(trunc):] = trunc

    return x

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
                        help='Input trainning data')
    parser.add_argument('--test', type=str,
                        default='../data/WikiNews/wikinews_test_lemma_stopwords.json',
                        help='Input testing data')
    parser.add_argument('--crossvalidation', type=str,
                        default='../data/WikiNews/wikinews_cv0.csv',
                        help='Crossvalidation data ids')
    parser.add_argument('--embeddings', choices=['self', 'fasttext'],
                        default='fasttext',
                        help='word embeddings representation')
    parser.add_argument('--train_embedding', action='store_true',
                        help='Add to train embeddings')
    parser.add_argument('--embedding_size', type=int,
                        default=256,
                        help='Embeddings layer size, when "embeddings=self"')
    parser.add_argument('--convs', type=int,
                        default=256,
                        help='Convolutional layer size')
    parser.add_argument('--dout', type=float,
                        default=0.1,
                        help='Drop out rate')
    parser.add_argument('--checkpoint', type=str,
                        default="checkpoint.pt",
                        help='checkpoint output file')
    parser.add_argument('--tune', action='store_true',
                        help='Add to train the model')
    parser.add_argument('--max_epochs', type=int,
                        default=25,
                        help='Maximum trainning epochs')
    parser.add_argument('--patience', type=int,
                        default=4,
                        help='Number of iterations to wait before stopping training when there is no error reduction')
    parser.add_argument('--logfile', type=str,
                        help='Output log file')

    args = parser.parse_args()
    setlog(args.logfile)
    logger.info(args)

    vocabulary_size = 2000
    sentence_length = 200

    x_data, y_data, _, labels, word2idx = load_data(args.train, vbsize=vocabulary_size)
    x_test, y_test, _, _, _ = load_data(args.test, word2idx)

    # cross-validation
    with open(args.crossvalidation, 'r') as cvidsfile:
        content = cvidsfile.readlines()
    cvids = {int(l) for l in content}

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    for irow, (x, y) in enumerate(zip(x_data, y_data)):
        if irow in cvids:
            x_train.append(x)
            y_train.append(y)
        else:
            x_valid.append(x)
            y_valid.append(y)

    # Preprocess
    x_train = pad_sequence(x_train, maxlen=sentence_length, pad_char=PAD, pad_post=False)
    x_test = pad_sequence(x_test, maxlen=sentence_length, pad_char=PAD, pad_post=False)
    x_valid = pad_sequence(x_valid, maxlen=sentence_length, pad_char=PAD, pad_post=False)

    if args.embeddings != 'self':
        embeddings_arr = load_pretrainned_embeddings(word2idx)
    else:
        embeddings_arr = None

    cnn = MyCNN(nlbs=len(labels), emb_arr=embeddings_arr, sentlen=sentence_length, nwords=vocabulary_size,
                freeze_emb=(args.train_embedding is False), emb_dim=args.embedding_size, nconv=args.convs,
                dropout=args.dout)

    if args.tune:
        logger.info("Train")
        train(cnn, x_train, y_train, x_valid, y_valid, args.checkpoint, epochs=args.max_epochs, patience=args.patience)

    logger.info("Eval")
    cnn.load_state_dict(torch.load(args.checkpoint))
    logresults(cnn, x_test, y_test, labels)
