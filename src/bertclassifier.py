import argparse
import logging
import json
import os
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import pipeline
import numpy as np

from functools import partial
from src.common import print_eval, logger
from datetime import datetime
from os.path import basename


def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def get_labels(dataset):
    if isinstance(dataset, str):
        with open(dataset, 'r') as f:
            content = json.load(f)
    elif isinstance(dataset, dict):
        content = dataset
    else:
        raise ValueError()

    return content['labels']


def split_data(inputfile, crossvalidation, out_train, out_val):
    with open(inputfile, 'r') as trainfile:
        train = json.load(trainfile)

    with open(crossvalidation, 'r') as cvidsfile:
        content = cvidsfile.readlines()
    cvids = {int(l) for l in content}

    traindata = []
    validdata = []
    for irow, row in enumerate(train['data']):
        if irow in cvids:
            traindata.append(row)
        else:
            validdata.append(row)

    labels = train['labels']
    with open(out_train, 'w') as f:
        json.dump({
            'part': 'train',
            'data': traindata,
            'labels': labels
        }, f, indent=4)

    with open(out_val, 'w') as f:
        json.dump({
            'part': 'valid',
            'data': validdata,
            'labels': labels
        }, f, indent=4)


def finetune(trainfile, crossvalidation, basemodelfile, outfile, cppath, seed=42, truncate=200, cpsteps=1000,
             tmptrainfile="bert_train.json", tmpvalfile="bert_val.json"):
    split_data(trainfile, crossvalidation, tmptrainfile, tmpvalfile)

    # 1. Data
    traindata = load_dataset(
        'json',
        data_files={
            'train': tmptrainfile,
            'valid': tmpvalfile
        },
        field='data'
    )
    strlabels = get_labels(tmpvalfile)

    # 2. Preprocess
    tokenizer = AutoTokenizer.from_pretrained(basemodelfile, model_max_length=truncate, do_lower_case=False)
    tokenized = traindata.map(partial(tokenize_function, tokenizer), batched=True)

    # 3. Train
    # evaluation_strategy="steps" para executar a avaliação junto com o log. A avaliação é feita com eval_dataset.
    training_args = TrainingArguments(
        num_train_epochs=5,
        evaluation_strategy="steps",
        # evaluation_strategy="no",
        save_steps=cpsteps,
        logging_steps=cpsteps,
        output_dir=cppath,
        seed=seed
    )

    # Utilizando este método da huggingface uma camada é inserida no final da rede BERT.
    # A linear layer is attached at the end of the bert model to give output equal to the number of classes.
    model = AutoModelForSequenceClassification.from_pretrained(basemodelfile, num_labels=len(strlabels))

    # O treinamento realiza as predições antes de chamar "compute_metrics". Em eval_pred já estão as predicões e y_true
    # As predições são feitas sobre o conjunto de dados indicado em "eval_dataset"
    metric = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['valid'],
        compute_metrics=compute_metrics
    )

    # Funciona continuar um treinamento iniciado em outra máquina. Mesmo em casos GPU->CPU
    # Quando a versão é diferente é emitido um alerta. É melhor usar a mesma versão.
    # trainer.train("test_trainer/checkpoint_wikinews-5000")
    trainer.train()

    trainer.save_model(outfile)
    os.remove(tmptrainfile)
    os.remove(tmpvalfile)


def model_eval(args, truncate=200):
    # Load data
    with open(args.test, 'r') as f:
        test = json.load(f)
    labels = get_labels(test)

    # Model predict
    model = AutoModelForSequenceClassification.from_pretrained(
        args.tunedmodel,
        local_files_only=True,
        id2label={i: i for i in range(len(labels))}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.basemodel, model_max_length=truncate, do_lower_case=False)

    logger.info("Eval")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = classifier([s['text'] for s in test['data']], padding=True, truncation=True)

    # Evaluation
    # for i, r in enumerate(results[:50]):
    #     logger.info(r, valid['data'][i]['label'])

    y_true = [d['label'] for d in test['data']]
    y_pred = [d['label'] for d in results]

    print_eval(y_true, y_pred, labels)


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
                        default='../data/WikiNews/wikinews_train.json',
                        help='training data')
    parser.add_argument('--test', type=str,
                        default='../data/WikiNews/wikinews_test.json',
                        help='testing data')
    parser.add_argument('--crossvalidation', type=str,
                        help='Crossvalidation data ids')
    parser.add_argument('--basemodel', type=str,
                        default='neuralmind/bert-base-portuguese-cased',
                        help='Bert model file (output if finetune, input if evaluation)')
    parser.add_argument('--tunedmodel', type=str,
                        help='Bert model file (output if finetune, input if evaluation)')
    parser.add_argument('--finetune', action='store_true',
                        help='Add to fine tune model')
    parser.add_argument('--logfile', type=str,
                        help='Output log file')

    args = parser.parse_args()
    setlog(args.logfile)
    logging.info(args)

    if args.finetune:
        finetune(
            trainfile=args.train,
            crossvalidation=args.crossvalidation,
            basemodelfile=args.basemodel,
            outfile='cpbert/wikimodel',
            cppath='cpbert',
            cpsteps=200,
            seed=42
        )
    else:
        model_eval(args)
# finetune(trainfile, testfile, basemodelfile, outfile):
