# Winner at EACL 2021 Shared Task: Dravidian-Offensive-Language-Identification

## Overview
This is official Github repository of team **hate-alert** which ranked 1st, 1st and 2nd on the Tamil, Malayalam and Kannada respectively in the shared task on ```Dravidian-Offensive-Language-Identification``` at the [DravidianLangTech-2021(The First Workshop on Speech and Language Technologies for Dravidian Languages)](https://dravidianlangtech.github.io/2021/), part of the [EACL 2021](https://2021.eacl.org/) conference. 

## Authors: Debjoy Saha, Naman Paharia, Debajit Chakraborty
Social media often acts as breeding grounds for different forms of offensive content. For low resource languages like Tamil, the situation is more complex due to the poor performance of multilingual or language-specific models and lack of proper benchmark datasets. Based on this shared task ```Offensive Language Identification in Dravidian Languages``` at EACL 2021, we present an exhaustive exploration of different transformer models, We also provide a genetic algorithm technique for ensembling different models. Our ensembled models trained separately for each language secured the 1st position in Tamil, the 2nd position in Kannada, and the 1st position in Malayalam sub-tasks.


## Finetuned models now available on HuggingFace :hugs:
1. [Tamil](https://huggingface.co/Hate-speech-CNERG/deoffxlmr-mono-tamil)
2. [Kannada](https://huggingface.co/Hate-speech-CNERG/deoffxlmr-mono-kannada)
3. [Malayalam](https://huggingface.co/Hate-speech-CNERG/deoffxlmr-mono-malyalam)

## Sections
1. [System description paper](#system-description-paper)
2. [Task Details](#task-details)
3. [Reproducing results](#reproducing-results) 
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

## System Description Paper  
Our paper can be found [here](https://arxiv.org/abs/2102.10084).    

## Task Details
The goal of this task is to identify offensive language content of the code-mixed dataset of comments/posts in Dravidian Languages ( (Tamil-English, Malayalam-English, and Kannada-English)) collected from social media. The comment/post may contain more than one sentence but the average sentence length of the corpora is 1. Each comment/post is annotated at the comment/post level. This dataset also has class imbalance problems depicting real-world scenarios. This is a comment/post level classification task. Given a Youtube comment, systems have to classify it into Not-offensive, offensive-untargeted, offensive-targeted-individual, offensive-targeted-group, offensive-targeted-other, or Not-in-indented-language. To download the data and participate, go to the "Participate" tab.

The Datasets are given below
1. Tamil - [Dev]() [Test]()
2. Kannada - [Dev]() [Test]()
2. Malayalam - [Dev]() [Test]()

## Methodology
In this section, we discuss the different parts of thepipeline that we followed to detect offensive posts in this dataset
1. [Machine Learning Models](#machine-learning-models)
2. [Transformer Models](#transformer-models)
3. [Fusion Models](#fusion-models)

### Machine Learning Models
As a part of our initial experiments, we used several machine learning models to establish a baseline per-formance. We employed random forests, logistic regression and trained them with TF-IDF vectors.The best results were obtained on ExtraTrees Classifier (Geurts et al., 2006) with 0.70, 0.63 and 0.95 weighted F1-scores on Tamil, Kannada and Malay-alam respectively.

### Transformer Models
We fine-tuned different state-of-the-art multilingual BERT models on the given datasets. This  includes  XLMRoBERTa, multilingual-BERT, Indic BERT and MuRIL. We also pretrain XLM-Roberta-Base on the target dataset for 20 epochs using Masked Language Modeling, to capture the semantics of the code-mixed corpus.  This additional pretrained BERT model was also used for finetuning. In addition, all models were fine-tuned separately using unweighted and weighted cross-entropy loss functions. For training, we use [HuggingFace](https://huggingface.co/transformers/pretrained_models.html) with [PyTorch](https://github.com/pytorch/pytorch).

### Fusion Models
Convolution neural networks are able to capture neighbourhood information more effectively. One of the previous state-of-the-art model to detect hatespeech was CNN-GRU (Zhang et al., 2018), We propose a new ```BERT-CNN``` fusion classifier where we train a single classification head on the concatenated embeddings from different BERT and CNN models. BERT models were initialised with the fine-tuned weights in the former section and the weights were frozen.  The number of BERT models in a single fusion model was kept flexible with maximum number of models fixed to three,due to memory limitation. For the CNN part, weuse the 128-dim final layer embeddings from CNNmodels trained on skip-gram word vectors usingFastText (Bojanowski et al., 2017)10. FastText vec-tors worked the best among other word embeddingslike LASER (Artetxe and Schwenk, 2019). For the fusion classifier head, we use a feed-forward neural network having four layers with batch normalization (Ioffe and Szegedy, 2015) and dropout (Srivas-tava et al., 2014) on the final layer. The predictionswere generated from a softmax layer of dimension equal to the number of classes.

Our pipeline is shown below
![Transformer Architecture](https://github.com/Debjoy10/Hate-Alert-DravidianLangTech/blob/master/architecture.png)


## Results  
Results of different models on the dev and test dataset can be found here:
The results have been in terms of the **Weighted-F1** scores.  

### Weighted  F1-score  comparison  for  trans-former,  CNN  and  Fusion  models: 

<h4 align="center">

|   Classifiers |Tamil | Tamil| Kannada | Kannada | Malayalam | Malayalam|
|---------------|------|------|------|------|------|------|
|               | **Dev** | **Test** |  **Dev** | **Test** |**Dev**| **Test** |
| XLMR-base (A) | 0.77 | 0.76 | 0.69 | 0.70 | 0.97 | 0.96 |
| XLMR-large    | **0.78** | **0.77** | 0.69 | 0.71 | **0.97** | **0.97** |
| XLMR-C (B)    | 0.76 | 0.76 | **0.70** | **0.73** | **0.97** | **0.97** |
| mBERT-base (C)| 0.73 | 0.72 | 0.69 | 0.70 | 0.97 | 0.96 |
| IndicBERT     | 0.73 | 0.71 | 0.62 | 0.66 | 0.96 | 0.95 |
| MuRIL         | 0.75 | 0.74 | 0.67 | 0.67 | 0.96 | 0.96 | 
| DistilBERT    | 0.74 | 0.74 | 0.68 | 0.69 | 0.96 | 0.95 | 
|  CNN          | 0.71 | 0.70 | 0.60 | 0.61 | 0.95 | 0.95 |
| CNN + A + C   | 0.78 | 0.76 | 0.71 | 0.70 | **0.97** | **0.97** |
| CNN + A + B   | **0.78** | **0.77** | 0.71 | 0.71 | **0.97** | **0.97** |
| CNN + B + C   | 0.77 | 0.76 | **0.71** | **0.72** | **0.97** | **0.97** |

</h4>

### Final Results:  
| Model Sets   | Tamil | Tamil| Kannada |Kannada| Malayalam | Malayalam     |
|--------------|-------|------|---------|------|-----------|------|
|              | Dev   | Test | Dev     | Test | Dev       | Test |
| Transformers | 0.80  | 0.78 | 0.74    | 0.73 | 0.98      | 0.97 |
| F-models     | 0.79  | 0.77 | 0.73    | 0.73 | 0.98      | 0.97 |
| R-models     | 0.79  | 0.78 | 0.74    | 0.74 | 0.97      | 0.97 |
| Overall      | **0.80**  | **0.78** | **0.75** | **0.74** | **0.98** | **0.97** |

Our Final Results on test dataset are given below
| Metrics   | Precision | Recall | Weighted-F1 | Rank |
|-----------|-----------|--------|-------------|------|
| Tamil     | 0.78      | 0.78   | 0.78        | 1    |
| Kannada   | 0.76      | 0.76   | 0.74        | 2    |
| Malayalam | 0.97      | 0.97   | 0.97        | 1    |



## Reproducing Results  
Ensure the following directory structure:

```bash
├── Finetuning/
├── Random_seed_ensemble/
├── CNN_embeddings/
├── ML Models/ 
├── README.md
└── LICENSE
```

### Citations
Please consider citing this project in your publications if it helps your research.
```
@inproceedings{saha-etal-2021-hate,
    title = "Hate-Alert@{D}ravidian{L}ang{T}ech-{EACL}2021: Ensembling strategies for Transformer-based Offensive language Detection",
    author = "Saha, Debjoy and Paharia, Naman and Chakraborty, Debajit and Saha, Punyajoy and Mukherjee, Animesh",
    booktitle = "Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages",
    month = apr,
    year = "2021",
    address = "Kyiv",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.dravidianlangtech-1.38",
    pages = "270--276",
    abstract = "Social media often acts as breeding grounds for different forms of offensive content. For low resource languages like Tamil, the situation is more complex due to the poor performance of multilingual or language-specific models and lack of proper benchmark datasets. Based on this shared task {``}Offensive Language Identification in Dravidian Languages{''} at EACL 2021; we present an exhaustive exploration of different transformer models, We also provide a genetic algorithm technique for ensembling different models. Our ensembled models trained separately for each language secured the first position in Tamil, the second position in Kannada, and the first position in Malayalam sub-tasks. The models and codes are provided.",
}
```


## Acknowledgements    

Additionally, we would like to extend a big thanks to the makers and maintainers of the excellent [HuggingFace](https://github.com/huggingface/transformers) repository, without which most of our research would have been impossible.
