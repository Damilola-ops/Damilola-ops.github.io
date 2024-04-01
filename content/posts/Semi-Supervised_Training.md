---
title: "Semi-Supervised training with BERT"
date: 2023-08-15T04:14:46+01:00
draft: true
cover:
    image: "molecule.gif"
    alt: ''
tags: ['BERT','Semi-supervised', 'Transformers']
Categories: ['NLP']
---

Learning to classify tweets using semi-supervised learning. 

Semi-Supervised learning is a training process typically suited for scenarios in machine learning with low amount of labelled samples and large amount of unlabelled samples. The goal of semi-supervised learning is to use an automated process that takes the information learned from the labelled samples to autonomously learn from the unlabelled ones.

In this article, we would attempt to teach a bert model to classify tweets into emotions and sentiments from 500 labelled samples and 25000! unlabelled samples and see how well we can get our model to perform. 
