---
title: "Byte-Pair Encoding, The Tokenization algorithm powering Large Language Models. "
date: 2023-07-20T04:14:46+01:00
draft: false 
cover:
    image: "tokenizers.jpg"
    alt: 'tokenizers'
tags: ['Natural Language Processing','Deep Learning', 'Transformers']
Categories: ['Machine Learning']
---


#  

Tokenization is an umbrella term for the methods used to turn texts into chunks of words or sub-words. Tokenization has a lot of applications in computer science, from compilers to Natural Language Processing. In this article, we would be focusing on tokenizers in Language models, in particular, a method of tokenization called Byte Pair Encoding.
The last few years have witnessed a revolution in NLP catalyzed mainly by the introduction of the transformers architecture in 2017 with the paper 'Attention is all you need ' epitomized by the introduction of ChatGPT in late 2022. 

## Why do we  tokenize texts and not just feed raw bytes to language models?

A tokenizer is a program that breaks text into smaller units that can be recognized by a model. In some deep learning models, inputs are fed in their raw forms as binary data such as images for CNNS and audio files for audio models.This makes sense as these models can learn the important features from their data in these formats. Computer vision models can learn features such as edges, textures and even encode translational invariance and equivariance. Language models are trained on text for them to able to learn semantic and syntactic relationships and patterns and by processing texts rather than bytes, these models can understand the linguistic structure of human language.

Text needs to be broken down into smaller units before they are passed into the layers of a language model and to understand why, we look at an interesting analogy between the human brain and deep learning models. In humans, babies learn to talk before even recognizing letters or how to spell words in most cases, and when we eventually start to learn the alphabet and how to spell words, we would have already built an intuitive understanding of the syntax and semantics of the language. For a language model, the story is quite different. Before training, the language model has no in-built knowledge of the language and has to learn the syntax and semantic relationships of language during training as it is fed petabytes of text during training. While we could attribute the reason for this to just an architectural choice and say 'nothing is stopping us from adding some form of rule-based system that adds this knowledge to the model before training, the choices we have made in the construction of transformers and large language models are for good reason.

Furthermore teaching a tokenizer how to break words and punctuations would require a large lookup dictionary containing all the possible patterns of word arrangement in the language, both during creation and look-up.

The act of tokenization predates Natural Language Processing in computer science and can be traced as far back as the early days of computing with the first compilers. In the process of converting high-level to machine language, the source code (usually raw ASCII or Unicode characters ) has to be broken down into chunks that are recognizable by the compiler called tokens that are usually language-specific keywords such as ' if ' or ' for', these tokens are then used to build a syntax/parse tree, the details of which are beyond the scope of this article and then passed down to other processes involved in execution.

##How does tokenization work?
A tokenizer is a program used to carry out tokenization
Given a sequence of text, a tokenizer turns the given sequence of text into a bunch of tokens. The tokens can be words, sub-words, or characters depending on the method of tokenization and the problem at hand.
The block of code below shows how to tokenize a sentence of text using the HuggingFace library.
<image is coming >
As you probably already figured out, there are multiple ways to go about this.

### A BRIEF HISTORY OF BYTE PAIR ENCODING

Byte Pair Encoding ( first described by Philip Gage in 1994) finds its roots in data compression. The Byte Pair Encoding algorithm is essentially a data compression algorithm that replaces the most frequently occurring pair of adjacent bytes with a new previously unused byte, recursively. If we have a string 'aaabcdaadab'. During the first iteration, the pair 'aa' is the most frequently occurring so it gets replaced by a new byte that we can call X . Our string is now 'XabcdXdab'. During the next iteration, the most frequently occurring pair is 'ab', so we replace it with a new byte 'Z' and we our string becomes 'XZcdXdaZ '. This process continues recursively till no further compressions can be made (every pair occurs only once). Decompression is simple and is done with a look-up table that contains the byte pairs and their replacements.

## Parallels between Tokenization and Compression

In Tokenization, the compression algorithm is modified to replace the most frequently occurring words as individual tokens, and the less frequently occurring words are broken into their more frequently occurring sub-words. For example the word ' compressor' would likely be split into 'compress' and 'or', since the former is likely to have occurred as a standalone word and the latter a suffix to many other words. From a data compression viewpoint, the byte 'compressor' has been broken down into two smaller bytes ' compress' and 'or' which is sort of the reverse of what we want during compression.
During tokenization, all unique tokens are stored in the tokenizer's vocabulary ( the tokenizer's equivalent of a look-up table) and the process of creating the vocabulary is typically referred to as ' training '.

## Tokenization

The tokenization process can be split into 2 steps, the training step and the actual tokenization of the input text. Training the tokenizer builds the tokenizer's vocabulary. One of the main advantages of byte pair encoding is that the tokenizer could be adapted for any corpus of interest and even languages as long as the smallest unit of the corpus can be encoded as bytes (irrespective of format - Unicode or ASCII). Typically tokenizers are trained on collections of datasets that encompass their use case. For example, if you are creating a tokenizer that is going to be used on medical problems, then you have to train it on a dataset containing medical terminologies and not everyday vocabulary, this way, the tokenizer can encode meaningful patterns such as prefixes like 'gastro', 'neuro' and other medical nuances that have specific meanings and we end up with tokens with meaningful word-level representations that can be learned by models during training.

The Training Process
Training a tokenizer isn't the same as training processes we are familiar with in machine learning and deep learning. Here, there are no gradients or optimizers as it is simply an encoding process where the tokenizer learns to create merges and learn new merge rules. At every step of the training process, the most pair of consecutive tokens are merged and the vocabulary is updated. This process is repeated until a specified vocabulary size is reached
Implementation
We would start by creating the example corpus or dataset we want our tokenizer to encode. Usually, this is replaced by an actual dataset like Wikipedia pages or books containing words that encompass the distribution of words we expect the tokenizer to come across. For simplification, our dataset is just a list of sentences (Tyler the Creator song titles if you haven't noticed).

```
corpus = ['the boy is a gun',
          'are we still friends',
          'sorry not sorry',
          'I thought you wanted to dance',
          'Wilshire',
          'exactly what you run from you end up chasing',
          'see you again ',
          'November',
          'Enjoy right now,today'
          'running out of time',
          ' '
          ]
```
The first step of the encoding process involves creating a base vocabulary of tokens for our tokenizer.
```
def create_base_vocab(corpus):
  base_vocab = [] 
  for sent in corpus:
    for char in sent:
      if char not in base_vocab :
       base_vocab.append(char)
  return base_vocab
```
Our base vocabulary consists of just characters of the alphabet and some other characters that occur in our dataset.

```
[' ',
 ',',
 'E',
 'I',
 'N',
 'W',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'l',
 'm',
 'n',
 'o',
 'p',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y']
```
Next, we create a dictionary that stores the frequency of occurrence of each word in our corpus. These word frequencies are used to calculate the frequency of byte pairs
```
from collections import defaultdict 

word_freq = defaultdict(int)
for sent in corpus:
  for word in sent.split():
    if word in word_freq.keys():
      word_freq[word] += 1 
    else:
      word_freq[word] = 1
```
We also create a dictionary of splits where each word in the dataset is tokenized on a character-level
` splits = {word:[char for char in word ] for word in word_freq.keys()}`
Some of our splits at this stage
```
{'the': ['t', 'h', 'e'],
 'boy': ['b', 'o', 'y'],
 'is': ['i', 's'],
 'a': ['a'],
 'gun': ['g', 'u', 'n'],
 'are': ['a', 'r', 'e'],
 'we': ['w', 'e'],
 'still': ['s', 't', 'i', 'l', 'l'],
 'friends': ['f', 'r', 'i', 'e', 'n', 'd', 's'],
 'sorry': ['s', 'o', 'r', 'r', 'y'],
```
The next step involves finding the most occurring byte pair. To do that, we need to find the frequency of all byte pairs in our corpus.
NB: Byte pairs are contiguous pair of tokens or characters

```
def find_pair_freqs():
  pair_freq = defaultdict(int)
  for word,freq in word_freq.items():
    split = splits[word]
    if len(split) < 1:
      continue 
    for i in range(len(split)-1):
      pair = (split[i],split[i+1])
      pair_freq[pair] += freq  
  return pair_freq
```
We find byte pair frequencies by iterating through every word in our corpus and creating byte pairs. The frequency of any byte pair is the same as the frequency of every word it occurs in. The pair frequency dictionary looks something like
``` 
defaultdict(int,
            {('y', 'o'): 4,
             ('w', 'a'): 1,
             ('a', 'n'): 2,
             ('n', 't'): 1,
             ('t', 'e'): 1,
             ('e', 'd'): 1,
             ('t', 'o'): 2,
             ('d', 'a'): 2,
             ('n', 'c'): 1,
             ('c', 'e'): 1,
             ('W', 'i'): 1,
```
we can now use our pair-frequency dictionary to find the most frequently occurring byte pair
```
#finding the most frequent pair 

best_pair = ''
max = None
for pair,freq in pair_freq.items():
  if max == None or freq > max :
    max = freq
    best_pair = pair
best_pair,max
```
the output of the code block above returns the most frequently occurring byte pair so far 

` (('o', 'u'), 6) `

Next, we write a function that takes the most prevalent byte pair and merges them.
 
```
def merge(a,b,splits):
  for word in word_freq.keys():
    split = splits[word]
    if len(split) == 1 :
      continue 
    i = 0  
    while i < len(split) - 1:
      if split[i] == a and split[i+1] == b :
        split = split[:i] + [a+b] + split[i+2:]
      else:
        i += 1 
    splits[word] = split
  return splits
```
We would expect the merge function to merge the tokens 'o' and 'u' since they are the most frequently occurring pair so far. Running our merge and viewing our splits

```
'thought': ['t', 'h', 'ou', 'g', 'h', 't'],
```
putting it all together, we repeat the steps above till we reach a preset vocabulary size.
```
vocab_size = 60
while len(vocab) < vocab_size:
  pair_freq = find_pair_freqs()
  best_pair = ''
  max = None
  for pair,freq in pair_freq.items():
    if max == None or freq > max :
      max = freq
      best_pair = pair
  splits = merge(*best_pair,splits=splits)
  merges[best_pair] = best_pair[0] + best_pair[1]
  vocab.append(best_pair[0]+best_pair[1])
  ```
Our vocabulary now contains all the new merges and tokens.

Vocabulary before Byte Pair Encoding  
 ```
 vocab 
[' ',
 ',',
 'E',
 'I',
 'N',
 'W',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'l',
 'm',
 'n',
 'o',
 'p',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y']
 ```
Vocabulary after Byte Pair Encoding
```
'I',
 'W',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'l',
 'm',
 'n',
 'o',
 'p',
 'r',
 's',
 't',
 'u',
 'w',
 'x',
 'y',
 'is',
 'th',
 'un',
 'en',
 'you',
 'an',
 'ex',
 're',
 'ti',
 'fr',
 'end',
 'so',
 'sor',
 'sorr',
 'sorry',
 'exa',
 'ha',
 'run',
 'in',
 'ing',
 'the',
 'bo',
 'boy',
 'gun',
 'are',
 'we',
 'sti',
 'stil',
 'still',
 'fri',
 'friend',
 'friends',
 'no',
 'not',
 'thou']
```

## Tokenization
The new vocabulary and merges can now be used to tokenize any input text. Notice that our base vocabulary consists of some alphabets and characters, if we try to use our tokenizer on text containing characters not present (X or z for example) in our vocabulary, our tokenize would raise an error. A simple solution is to simply define all the letters of the alphabet or even better, use all ASCII characters as our vocabulary. For the GPT models, OpenAI uses a method known as byte-level byte pair encoding, instead of alphabets or ASCII, the base vocabulary is defined in bytes. Since every character in any encoding on a computer is created from bytes, the base vocabulary contains every possible byte, and the tokenizer never runs into an unknown token.

## Why Byte Pair Encoding?
The advantages of byte pair encoding are not immediately apparent and to understand them, we would discuss the common problems that are presented when we try to tokenize a dataset for a language model to see why byte pair encoding is the tokenization algorithm of choice in GPT3 and other language models.
Unknown words, Large Vocabularies, Information per token
If we chose to tokenize words by any other means, say splitting by white space, for example, the first problem we run into is an extensive vocabulary. Tokenizing by whitespace means that our tokenizer would have to encode the words 'fast', ' faster ', and ' fastest ' individually. This means words and their tenses and affixes are encoded differently, thereby losing meaningful semantic relationships between these words as easily. Specifically every word and its [whatever prefixes and suffixes are called ] are encoded individually and this leads to us having a much larger vocabulary and more computational costs during model training and inference. With byte pair encoding, words 'fast' and 'er' are treated as two different subwords, and hence every occurrence of faster would be encoded as 'fast' and '##er', with the double hashtag indicating that 'er' is a suffix that has to be merged with 'fast'. This is because our tokenization algorithm would learn that 'er' is a subword that occurs quite frequently since it occurs in the superlative of most words and would eventually treat it as a word on its own. So the word 'faster' would be encoded as ['fast', '##er'] and the same applies to ' fastest '. As we can see, BPE allows us to reduce our vocabulary size without losing too much information per token. We get to keep the semantic information that word-level tokenization provides without incurring the cost of an extensive vocabulary.
The second problem that arises is in recognizing rare words. If a word occurs in our text that our word-based tokenizer doesn't recognize, it would have to be replaced by a [UNK] token signifying that the token is unknown and the tokenizer doesn't recognize that word. With Byte Pair Encoding, unknown words would be split into smaller known sub-words that exist in the vocabulary and merged.
The final and most important problem arises from  the amount of information per token a tokenizer encodes. A character-level tokenizer for example would break every word into a sequence of characters, and while we eliminate the problem of unknown words, a language model would not be able to learn meaningful semantic relationships if all it takes as input is a sequence of characters. Imagine if instead of seeing an image, we are instead given a sequence of individual pixel values and told to figure out what the image entails. Byte pair encoding finds the balance between character-level tokenization and word-level tokenization without losing relevant information about words.

