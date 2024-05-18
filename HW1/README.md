# HW1: Persian Language Model with SRILM

## Purpose

The purpose of this assignment is to:

1. Collect a Persian dataset in a specific domain with at least 20,000 words.
2. Extract unigram, bigram, and trigram language models.
3. Apply different smoothing approaches to these models.
4. Calculate perplexity on another Persian dataset within the same domain, containing at least 2,000 words.


## Dataset
For this project, I utilized the [Hamshahri newspaper dataset](https://dbrg.ut.ac.ir/hamshahri/), which includes various news categories from 1996 to 2007. This dataset was collected at the University of Tehran.

## Selected Subset
Year: 2007

Category: Social

Word Count: 77,758 words

## Preprocessing
The dataset was preprocessed using the Parsivar library to ensure the text was clean and ready for language modeling.

## Tools
For creating language models, I used the SRILM (SRI Language Modeling Toolkit) with linux commands.

## Language Models and Smoothing Techniques
Three types of n-gram models were created: Unigram, Bigram and Trigram

For each model, the following smoothing approaches were applied: Laplace Smoothing, Absolute Discounting, Kneser-Ney Smoothing



## Perplexity Calculation
| model \ smothing |  Add one(Laplace) | Discounting Constant (Absolute) | Kneser-Kney |
| -------- | ------- | ------- | ------- |
| unigram | 1568.506 | 1535.483 | 1547.086 |
|  bigram | 1569.118 | 453.2217 | 384.1172 |
| trigram | 1680.4 | 429.9058 | 361.3825 |











