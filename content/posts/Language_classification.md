---
title: "Classifying Programming Languages from short code snippets  "
date: 2023-08-10T04:14:46+01:00
draft: false 
cover:
    image: ""
    alt: ''
tags: ['Natural Language Processing','Deep Learning', 'Transformers']
Categories: ['Machine Learning','NLP']
---

# 

This is a simple classification problem where we try to build a  classifier to classify code snippets into the programming language it was written in. 
We would start with simple machine learning approaches and work our way up to more complex methods till we have a satisfactory solution. 

## The Dataset 
Our dataset is a csv containing 45,000 samples . The dataset is made up of two columns, the 'code' feature contains  code snippets we want to classify and the language column, which is our label contains the programming language it belongs to.Our train and test datasets were created from stratified sampling based on the target variable. 

# Exploring the dataset 
To get a better picture of our dataset , we look at the distribution of classes in the dataset 


We also check the number of unique categories in our label . 


## Data Cleaning 
We don't have to do too much data cleaning, the nature of our problem suggests we keep our input just the same , since we are trying to learn the syntax of different languages, it is preferrable to keep the code snippets untouched and hope our model picks up hidden nuances and signals.

## Creating a Baseline solution 

Our first model would be a multinomial Naive Bayes classifier. For preprocessing our text, we would try  a count vectorizer and tf-idftransformer. 
We would use the sklearn library's implementation of the aforementioned algorithms. 

## Using BERT as a feature extractor
Our first encounter with transformers would be to use them as feature extractors.Using BERT as a feature extractor means using the model hidden states produced in the last layer as features that would be used to train a classifier. The hidden states are simply context-enriched embeddings (a 768-dimensional tensor) produced by multiple self=attention layers in BERT. We would then use this hidden states to train a simple classifier like a Logisitic-Regressor or a Random-Forest. I decided to go with a Logistic Regressor. After fitting the model, to our hidden states and trying to make predictions , our model performs better than the naive baiyes classifier with an accuracy of ' ', a step in the right direction but definitely not where we would like to be if we want a model that knows how to classify languages  


## Finetuning BERT 
In the spirit of progressively increasing complexity, I have decided to jump the gun and just skip to the state of the art . We would be using the BERT-base model with a classification head(a fully connected layer with pooling applied) to try and solve the problem . 

In the first training run , I decided to finetune BERT for only 5 epochs , with a max_token_length of 512 and using 16-bit floating point numbers for the model's weights . 

As expected, the BERT model perfomance was significantly better than the previous two models we tried with an accuracy of 90% and an F1 score of 89 . Great, but we still not good enough . An obervation was I made when I tried handcrafted code samples was that the model was very good at recognizing python and javascript code, but  struggled with 'R' and Scala. This is explainable by the fact that our training dataset consists of only 127 examples of R and 270 examples of Scala, the model had probably not seen enough R or Scala during training . 

During the final run, I trained for 10 epochs using the same training parameters as before and saw a ''% accuracy and an F1 score of 90

## An Intersting side note 
An interesting problem arises when we try to read our data and tokenize. Since our dataset consists of code snippets that were crawled from the internet, some rows of our dataset contain buggy lines such as unclosed curly brackets for example. The problem with this is that when pandas or any csv parser tries to parse the strings of our  dataset and runs into an unexpected EOF character such as an unclosed curly bracket or quotation , since csv parsers rely on balanced structures, unclosed quotations will break the parsing context and cause the parser to raise an EOF error . To work around this , I decided to replace all EOF characters ("/x1A") in ASCII as part of the preprocessing and tested the model predictions to see if valuable signals or information where not lost. Another workaround is to use the argument ` error_bad_lines=False` when reading the dataset 




