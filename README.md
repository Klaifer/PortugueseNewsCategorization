# Portuguese News Classification

Code for PLOS ONE 2024 paper [Breaking News: Unveiling a New Dataset for Portuguese News Classification and Comparative Analysis of Approaches](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296929)

# Installation

1. Save a copy of this repository;
2. Install the packages in requirements.txt;
3. Download spaCy's Portuguese language trained pipeline (pt_core_news_lg);
4. Download and unzip the fasttext in "src", pretrained in Portuguese [(cc.pt.300.bin.gz)](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pt.300.bin.gz);
5. Install DJINN [(github)](https://github.com/LLNL/DJINN)


# Data

This code uses datasets in json format with the following configuration:
```
    {
        "part": "test|train",
        "data": [
            { 
                "pageid": number,
                "text": "... content ...",
                "label": index on labels list
            },
            ...
        ],
        "labels": [
            "class_label1",
            "class_label2",
            ...
        ]
    }
```

Data produced by [WikiNews source](https://github.com/Klaifer/PortugueseNewsDataset) is generated in this format.

Additionally, we include the article ids in data/FolhaSP to replicate the separation into training and test sets.

# Preprocessing

## Vocabulary preprocessing
To compose the dictionaries used by the classifiers, we preprocessed the sentences.
In this preparation the words are changed to their canonical (non-inflected) form, lowercase and stop-words are removed.
```
    python preprocess_text.py \
           --input ../data/WikiNews/wikinews_train.json \
           --output ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --operation lemma \
           --stopwords

    python preprocess_text.py \
           --input ../data/WikiNews/wikinews_test.json \
           --output ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --operation lemma \
           --stopwords
```

## Sentence embeddings

We produce datasets with sentence embeddings.
These same data produced are applied in several experiments.
```
    python preprocess_fasttext.py \
           --input ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --output ../data/WikiNews/wikinews_train_lemma_stopwords.fasttext.json
    
    python preprocess_fasttext.py \
       --input ../data/WikiNews/wikinews_test_lemma_stopwords.json \
       --output ../data/WikiNews/wikinews_test_lemma_stopwords.fasttext.json
```

## Cross-validation

For the most accurate comparison, it is preferable that all methods run on the same data partitions.

The way we chose to guarantee this separation running on different equipments is the use of files.

So, we produce one file per partition, each file contains a list of numbers, which correspond to the 
indices of the input texts that must participate in the training set. The remainder should be the 
validation set.

The files are created with the following command. 
The application of the scripts is encoded in the different methods.
It can be applied over preprocessed or raw input file, as only class labels are used.
```
    python preprocess_crossvalidation.py \
           --train ../data/WikiNews/wikinews_train.json \
           --nparts 5 \
           --output_prefix ../data/WikiNews/wikinews
```

# Classification
In the following examples we use the first cross-validation partition. Commands can be changed to include others.

## Support Vector Machine + BoW

```
    python svm_words.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --bow \
           --logfile ../logs/svm_bow_wiki_cv0.log
```

## Support Vector Machine + Tf-IDF

```
    python svm_words.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --logfile ../logs/svm_tfidf_wiki_cv0.log
```

## Support Vector Machine + Sentence embedding (fastText)
```
    python svm_features.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.fasttext.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.fasttext.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --logfile ../logs/svm_fasttext_wiki_cv0.log
```

## CNN: Topology definition

With the following script, different configurations of the CNN network are evaluated.
The results can be used to select the best configuration.
```
    python optimizer_cnn.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.fasttext.json \
           --logfile ../logs/cnn_optimize_wiki.log
```

## CNN + Pretrained Embeddings
```
    python cnnclassifier.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --embeddings fasttext \
           --train_embedding \
           --convs 64 \
           --dout 0.1 \
           --tune \
           --logfile ../logs/cnn_fasttext_wiki_cv0.log
```

## CNN + Self-trained embeddings

```
    python cnnclassifier.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --embeddings self \
           --train_embedding \
           --embedding_size 256 \
           --convs 128 \
           --dout 0.2 \
           --tune \
           --logfile ../logs/cnn_selfemb_wiki_cv0.log
```

# DJINN + Bow

```
    python djinn_words.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --bow \
           --ntrees 50 \
           --max_depth 4 \
           --logfile ../logs/djinn_bow_wiki_cv0.log
```

# DJINN + TF-Idf
```
    python djinn_words.py \
           --train ../data/WikiNews/wikinews_train_lemma_stopwords.json \
           --test ../data/WikiNews/wikinews_test_lemma_stopwords.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --ntrees 50 \
           --max_depth 4 \
           --logfile ../logs/djinn_bow_wiki_cv0.log
```

# BERT

The BERT experiments were done in two steps. In the first the training and in the second the evaluation.

In training, the inputs are the training data and the description of the cross-validation part. During the
evaluation is used only the test data.

During training, the error observed in the validation set (which was extracted from the training data) 
is displayed. 
All checkpoints are saved. Note the checkpoint with the lowest error in the validation set. This is the 
best partition to be used in the evaluation stage (expectation of better generalization).

Training can be done with the following command:
```
    python bertclassifier.py \
           --train ../data/WikiNews/wikinews_train.json \
           --crossvalidation ../data/WikiNews/wikinews_cv0.csv \
           --finetune \
           --logfile ../logs/bert_train_wiki_cv0.log
```

The application can be done with the following command:
```
    python bertclassifier.py \
           --test ../data/WikiNews/wikinews_test.json \
           --tunedmodel cpbert/checkpoint-800 \
           --logfile ../logs/bert_eval_wiki_cv0.log
```

# Citation

If you find this code useful, please cite:
```
@article{10.1371/journal.pone.0296929,
    doi = {10.1371/journal.pone.0296929},
    author = {Garcia, Klaifer AND Shiguihara, Pedro AND Berton, Lilian},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Breaking news: Unveiling a new dataset for Portuguese news classification and comparative analysis of approaches},
    year = {2024},
    month = {01},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0296929},
    pages = {1-15},
    number = {1},
}
```
