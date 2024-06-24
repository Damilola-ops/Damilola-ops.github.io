---
title: "The untold problems that arise when Training Byte-Pair-Encoding Tokenizers on large datasets"
date: 2024-02-15T04:14:46+01:00
draft: True 
cover:
    image: ""
    alt: ''
tags: ['Natural Language Processing', 'Transformers']
Categories: ['NLP','LLM']
---


The first step to training a language model on a new domain is to train a tokenizer. In this article, we would be focusing on the challenges that arise when training tokenizers on large datasets, in particular, Byte-Pair-Encoding Tokenizers. 

If you weren't already familiar with Byte-Pair-Encoding, You can check out Karpathy's(link to karpathy BPE) video or an article I wrote a while ago[👀](https://damilojohn.github.io/posts/BPE)


Training a BPE tokenizer is a CPU and memory heavy tasks that can get out of hand pretty quickly. In this article, we would be trying to train a SentencePiece BPE tokenizer on a multilingual dataset with 13 million rows!!.

