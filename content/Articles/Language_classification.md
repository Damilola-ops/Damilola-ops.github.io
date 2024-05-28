---
title: "Teaching BERT to identify 15 programming languages. "
date: 2023-08-19T04:14:46+01:00
draft: false 
cover:
    image: "prog_class.jpg"
    alt: ''
tags: ['Natural Language Processing','Deep Learning', 'Transformers']
Categories: ['LLM','Finetuning']
---

# 

This is  a fun side project where I tried  to build a model that could identify 15 of the most popular programming languages.
We would start with simple machine learning approaches and gradually work our way up to more complex methods till we have a satisfactory solution. 

## The Dataset 
Our dataset is a csv containing 45,000 samples. The dataset is made up of two columns, the 'code' feature contains  code snippets we want to classify and the language column, which is our label contains the programming language it belongs to.Our train and test datasets were created from stratified sampling based on the target variable. 

## Exploring the dataset 
For a clearer picture of our dataset, let's take a look at the distribution of classes in the dataset 
![class distribution](https://proglangclassifier.s3.eu-west-2.amazonaws.com/class+distribution.png)

We also check the number of unique categories in our label . 


## Data Cleaning 
Initially, I started but trying to create a baseline perfomance with no data cleaning or preprocessing.Since we are trying to learn the nuances of different programming languages , I tried to keep the code just as is and see how the models performed before moving forward with any form of preprocessing or feature engineering.

## Creating a Baseline solution 

Our first model would be a multinomial Naive Bayes classifier. For preprocessing our text, we would try  a count vectorizer and tf-idftransformer. 
We would use the sklearn library's implementation of the aforementioned algorithms. 
![naive bayes](https://proglangclassifier.s3.eu-west-2.amazonaws.com/naive_baiyes.png)
A bit better than random guesses even when you factor in our heavy class imbalance.

## Finetuning BERT 
In the spirit of progressively increasing complexity, I have decided to jump the gun and just skip to the state of the art . We would be using the BERT-base model with a classification head(a fully connected layer with pooling applied) to try and solve the problem . 

In the first training run , I decided to finetune BERT for only 5 epochs , with a max_token_length of 512 and using 16-bit floating point numbers for the model's weights . 
![old accuracy score](https://proglangclassifier.s3.eu-west-2.amazonaws.com/old_accuracy_score.png)


As expected, the BERT model perfomance was significantly better than the previous two models we tried with an accuracy of 90% and an F1 score of 0.89 . Great, but we still not good enough . An obervation was I made when I tried handcrafted code samples was that the model was very good at recognizing python and javascript code, but  struggled with 'R' and Scala. This is explainable by the fact that our training dataset consists of only 127 examples of R and 270 examples of Scala, the model had probably not seen enough R or Scala during training . 

During the final run, I trained for 10 epochs using the same training parameters as before and saw a ''% accuracy and an F1 score of 90 

## Getting better perfomance and model optimization 
I started to think about ways to improve the model's perfomance by preprocessing  without losing too much useful information.I decided to look at some of the tokenizers outputs when I found out that the BERT BPE tokenizer doesn't have a token for represent '/n' and '/t', newline and tab characters, respectively. This meant that our model only saw an [UNK] token, which results in a lot of lost information as key programming syntax such as loops and conditionals are defined by both characters. As a workaround, I created new tokens in the tokenizer called [NEWLINE] and [TAB]. I also replaced replaced all instances of integers and floats in the code samples as those are useless anyway and replaced them with [FLOAT] or [INT]. 


Training on the new dataset gave an improved accuracy 92% and F1 score of 0.92 with  a smaller DISTILBERT model (BERT but with model distillation)
![distilbert training scores](https://proglangclassifier.s3.eu-west-2.amazonaws.com/acc_with_feature_processing.png)


## An Interesting side note 
An interesting problem arises when we try to read our data and tokenize. Since our dataset consists of code snippets that were crawled from the internet, some rows of our dataset contain buggy lines such as unclosed curly brackets for example. The problem with this is that when pandas or any csv parser tries to parse the strings of our  dataset and runs into an unexpected EOF character such as an unclosed curly bracket or quotation , since csv parsers rely on balanced structures, unclosed quotations will break the parsing context and cause the parser to raise an EOF error . To work around this , I decided to replace all EOF characters ("/x1A") in ASCII as part of the preprocessing and tested the model predictions to see if valuable signals or information where not lost. Another workaround is to use the argument ` error_bad_lines=False` when reading the dataset 

## Testing out the model 
After evaluating the model on an holdout set , our both metrics were still holding good. I decided to try out some code samples suggested by CHATGPT 
![code samples image](https://proglangclassifier.s3.eu-west-2.amazonaws.com/code_examples.png)
![result]()
And on python code I wrote myself, 
![handcrafted sample](https://proglangclassifier.s3.eu-west-2.amazonaws.com/handcrafted_example.png)

I noticed the model confused  Rust and C++ code . This could be explained by the fact that Rust and C++ have very similar syntax and are difficult to tell apart even for the average human.









