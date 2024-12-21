---
title:  "Descrambling Sentences with GPT2 "
date: 2024-06-10T04:14:46+01:00
draft: false 
cover:
    image: "anagrams.jpg"
    alt: ''
tags: ['Natural Language Processing', 'Transformers']
Categories: ['NLP','LLM']
---
## Finetuning GPT2 to Reconstruct sentences





Two words are anagrams if one can be formed by permuting the letters of the other. Applying the same logic to a sentence, would be saying that two sentences are anagrams(no such thing) if their component words can be permutated to form clones of each other.

I thought it would be interesting to finetune a language model to do this. You might be thinking that simply re-arranging words in a sentence doesn't require intelligence and can be done with very trivial algorithms,you would be right,  but I added an edge to this task, given a random sequence of words, the language model has to return a grammatically correct sequence using the same set of words. For example, the following sequence: 
  ```
  The equations expensive. show is optimization computationally that
  ```
 returns:
 ```
 The equations show that optimization is computationally expensive.
 ```
Link to [article](https://damilojohn.github.io/articles/text-desrambler/)

View code [here](https://github.com/damilojohn/Text-Descrambling)
and the model on [huggingface](https://huggingface.co/damilojohn/text-descrambler)