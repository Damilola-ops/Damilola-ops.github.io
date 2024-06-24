---
title: "Finetuning GPT2 to Reconstruct Sentences"
date: 2024-06-15T04:14:46+01:00
draft: false 
cover:
    image: "anagrams.jpg"
    alt: ''
tags: ['Natural Language Processing', 'Transformers']
Categories: ['NLP','LLM']
---

Two words are anagrams if one can be formed by permuting the letters of the other. Applying the same logic to a sentence, would be saying that two sentences are anagrams(no such thing) if their component words can be permutated to form clones of each other.

I thought it would be interesting to teach a language model to do this. You might be thinking that simply re-arranging words in a sentence doesn't require intelligence and can be done with very trivial algorithms,you would be right,  but I added an edge to this task, given a random sequence of words, the language model has to return a grammatically correct sequence using the same set of words. For example, the following sequence: 
  ```
  The equations expensive. show is optimization computationally that
  ```
 should return:
 ```
 The equations show that optimization is computationally expensive.
 ```

![guy from the office](https://images.aicrowd.com/uploads/ckeditor/pictures/457/content_giphy__8_.gif)

 More examples:
 ```
 'the which wiring flow. propose to diagram, method network a reflects signal We visualize'  ---> 'We propose a method to visualize the wiring diagram, which reflects network signal flow.'
 
 ,
 'the interaction networks. the gap Finally, analyze chemical the junction between synapse and we', --->  'Finally, we analyze the interaction between the gap junction and the chemical synapse networks.'


 'the process The pseudorandom number illustrated in is Mathematica. generator using' ---> 'The process is illustrated using the pseudorandom number generator in Mathematica.',

 
 'statistical estimators functionals. of various of consistent We investigate existence the bounded-memory' --->  'We investigate the existence of bounded-memory consistent estimators of various statistical functionals.'


 'rather negative sense. the question This in strong a is in resolved' ---> 'This question is resolved in the negative in a rather strong sense.'
 ```



In this article, I would be explaining  how I fine-tuned a small version of GPT-2 (334m parameters) on the task of turning descrambled sentences into their grammatically correct forms using the same words in the sentence. I would be exploring the different challenges and trade-offs that were made faced from data preparation and training optimizations, all the way to text-generation strategies

## Data

The dataset consists of 40000 rows of descrambled sentences and their labels, with columns ['text','id','label'].

Building the dataset involved scraping wikipedia pages and picking random sentences from them. This sentences were then permutted to create the 'text' - 'label' pairs used for training the language model. To ensure a reasonable data distribution, I scraped pages ranging from niche Medicine, engineering and mathematics pages to literature and even fandom pages. 

```
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'id'],
        num_rows: 40001
    })
    test: Dataset({
        features: ['text', 'label', 'id'],
        num_rows: 10000
    })
    val: Dataset({
        features: ['text', 'label', 'id'],
        num_rows: 4001
    })
})
```
The dataset consists of 54000 rows of sentences, split into train, test and validation.

## Data Preprocessing

One of the key challenges in finetuning GPT-2 for sentence reconstruction was in framing the problem in way that was learnable for the model. As a decoder-only model with causal mask attention, GPT-2 is trained to predict the next token using only the previous tokens in the sequence. This is fundamentally different from the typical sequence-to-sequence (seq2seq) approach, where the entire input sentence is used as context to predict each token in the target sentence.

In a standard sequence-to-sequence task, an encoder-decoder model like the one used in the original Transformer paper is the reasonable choice. The encoder processes the input sentence, and the decoder generates the output sentence by attending to the encoder's representations. This allows the model to directly leverage the input context when producing the target sequence.

However, with a decoder-only model like GPT-2, we need to rethink how we structure the dataset and the training process. The problem here lies in ensuring gpt2 'sees' the permutted form of the sentence and uses this to generate the output, this way, we can hope our model learns to use the previous incorrect sentence to generate a grammatically meaningful anagram of the input. 

Each sentence in the  training dataset was formatted to the following template:

```
Wrong sentence: {scrambled sentence}
Correct sentence: {descrambled sentence}
```
By providing the model with the full scrambled sentence and asking it to complete the task by outputting the correct, unscrambled version, we have essentially framed a sequence-sequence task as a text generation one.


```
def preprocess_data(row):
        target_text = row['label']
        # add prompt to every row in the dataset
        input_text = f'''wrong sentence: {row['text']} correct sentence:'''
        # find the length of the input prompt 
        prompt_len = len(tokenizer(input_text).input_ids)
        input = tokenizer(f'{input_text} {target_text} <|endoftext|>',
                          padding='max_length', truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
        input_ids, attention_mask = input.input_ids, input.attention_mask
        # turn all of the tokens before the actual correct sentence to -100
        # so loss is only calculated for generation after 'correct sentence:'
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        # Turn all pad tokens to -100
        labels[labels == tokenizer.pad_token_id] = -100
        assert (labels == -100).sum() > len(labels), "Labels are all -100,something is wrong."
        # if (labels == -100).sum() == len(labels) - 1:
        #         raise
        return {'input_ids': input_ids.squeeze(),
                'attention_mask': attention_mask.squeeze(), 
                'labels': labels.squeeze(),
                'prompt': input_text}
    processed_data = dataset.map(preprocess_data)
    processed_data.set_format(type='torch', columns=['input_ids',
                                                     'attention_mask', 'labels'
                                                     ])



processed_data = dataset.map(preprocess_data,batched=True,batch_size=256)
processed_data.set_format(type='torch',columns=['input_ids','attention_mask','labels'])
```

## The Preprocess Function

The first three lines of the function creates our prompt template and tokenizes every row in our dataset. Next we calculate the length of the prompt part of the template i.e " wrong sentence: {input text} correct sentence:" . We then concatenate the input and labels together to form our template and successfully turn our problem to a next word prediction problem.

Then set every token in the labels before 'correct sentence:' to -100 (Transformer's library doesn't calculate loss for this token and was chosen arbitrarily).
Why ?

The reason for doing this is to ensure that the model doesn't waste  steps learning how to predict the prompt part of the sentence and only learns to predict tokens that comes after the  'correct sentence:'. This was done to make sure that we are  backpropagating losses related to descrambling sentences only and not add any noise that might confuse the model further 

This is done by ensuring the model only calculates loss on the actual tokens we need to learn. -100 is  the ignore index by pytorch's cross_entropy_loss function and was chosen arbitrarily.

Finally the preprocessing is completed by also setting all pad tokens in the labels to -100 for the reason stated above, we don't want our model paying attention to padding tokens.

Here's a visual breakdown of the whole process
Sentence =
```

``` 
``` important and unmixing data challenging an problem hyperspectral Spectral in processing. is ``` --> Tokenizer --> input_ids : 
```
Tensor([36460,  6827,    25,   286,   318, 31760,  7468, 10393,   317,  2276,
         5545,    13,  3376,  6827,    25, 31760,  7468,   286,   317, 10393,
          318,  5545,    13,  2276,   220, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
        50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256])
```
with 50256 being the pad_token_ids.


input_ids  ---> Preprocess function --> 

```
tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100, 31760,  7468,   286,   317, 10393,
          318,  5545,    13,  2276,   220,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
         -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100])
```
every token but the tokens corresponding to everything after 'correct sentence' or the descrambled sentence is converted to -100 in the label, ensuring that loss is only calculated on this tokens.


## Hardware

For this task, the Nvidia A10G(24GB) provided by [modal](https://modal.com) was used for training, since GPT2-medium has 335m parameters and I trained in 16bit precision, adding up to about 5.6gb in memory to store model weights, and for training about twice that for training(optimizer states and gradients). Using a 16gb GPU with a reasonable batch size would have been sufficient, however training for shorter periods (larger batch sizes) on a  more expensive GPU turned out to be cheaper.
```
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
```

## Finetuning

Finetuning in modal was fairly straightforward from this point, all I had to do was create a docker-like image containing all of our needed libraries that runs in a sandboxed [gvisor container](https://cloud.google.com/blog/products/identity-security/open-sourcing-gvisor-a-sandboxed-container-runtime) and installed all our neccessary libraries.

Next we wrote functions such as our trainer class and other utils to be executed in the container environment we created.
For training, I used a learning rate of 3e-5 (5e-5 is typically used to train gpt2 but I tried a smaller learning rate since I used a larger batch size).
```
 training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        num_train_epochs=5,
        learning_rate=3e-5,
        output_dir=str(VOL_MOUNT_PATH / "model"),
        logging_dir=str(VOL_MOUNT_PATH / "logs"),
        logging_strategy='steps',
        logging_steps=100,
        load_best_model_at_end=True,
        save_strategy='steps',
        evaluation_strategy='steps',
        save_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        report_to='wandb',
        fp16=True
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_data['train'],
        eval_dataset=processed_data['validation'],
        )
    trainer.train()
    wandb.finish()
    model.save_pretrained(str(VOL_MOUNT_PATH / "model"))
    tokenizer.save_pretrained(str(VOL_MOUNT_PATH / "tokenizer"))
```
I trained for 5 epochs(actually the validation loss stopped improving after the 3rd epoch and I was a bit worried about overfitting as I didn't set any early stopping criteria)

![train loss](https://proglangclassifier.s3.eu-west-2.amazonaws.com/text-descrambler-train_loss.png)
![validation loss](https://proglangclassifier.s3.eu-west-2.amazonaws.com/eval_loss_text_Descrambler.png)



### Optimization 

Using some memory optimization training techniques for a single GPU like CPU offloading(moving optimizer states to main memory), gradient checkpointing and gradient accumulation, I was able to fit a batch size of 128 in memory. Training was also done in FP16.


## Inference

Now to the fun part, testing out the model.Testing the model with the same inputs displayed earlier in the article which were chosen randomly from the test set, Here are the outputs :
```
Output: wrong sentence: the which wiring flow. propose to diagram, method network a reflects signal We visualize correct sentence: ---> The diagram, which reflects the wiring network to propose a signal flow.                                                                                                                  
Output: wrong sentence: the interaction networks. the gap Finally, analyze chemical the junction between synapse and we correct sentence: --->The gap junction and the chemical interaction between the synapse networks.                                                                                                                   
Output: wrong sentence: the process The pseudorandom number illustrated in is Mathematica. generator using correct sentence:The pseudorandom number generator is illustrated using the process in Mathematica.                                                                                                                
Output: wrong sentence: in the of structure resulted decrease mutual signal in information. Introducing correlations input-output correct sentence: ---> This resulted in decrease of mutual correlations in the input-output structure of signal information.                                                                                                               
Output: wrong sentence: statistical estimators functionals. of various of consistent We investigate existence the bounded-memory correct sentence: ---> The existence of bounded-memory estimators consistent with various statistical functionals.                                                                                                                 
Output: wrong sentence: rather negative sense. the question This in strong a is in resolved correct sentence: --> The question is resolved in a rather strong negative sense.
```



By merely visually inspecting the outputs the first thing I noticed was how easy it was to make up sentences that looked correct but actually make no sense. For example:

```
differential low-power The comparators. fully uses ADC clocked
```
our finetuned model output:
```
Output: wrong sentence: differential low-power The comparators. fully uses ADC clocked correct sentence:The ADC uses fully clocked low-power differential comparators.
```


GPT3.5 :

![chatgpt's answer](https://proglangclassifier.s3.eu-west-2.amazonaws.com/Screenshot+(17).png)

ChatGPT's output makes no sense whatsoever as there's no such thing as a differential low power ADC

**Chatgpt was prompted using some few-shot examples from the training set.**


## Generation Strategy

For generation, I used beam-search as a decoding strategy as it felt like the most reasonable decoding strategy for this task since the sequence with the highest overall probability was more likely to be output I want.



In conclusion, the 335m parameter model finetuned on this task performs alot better than the 175b GPT3.5 with few-shot prompting. It was interesting to find that a 335m parameter model could learn to descramble sentences and generalize well on unseen samples.

View the code [here](https://github.com/damilojohn/Text-Descrambling)
and the model on [huggingface]()