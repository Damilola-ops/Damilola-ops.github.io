---
title: "Finetuning GPT2 to Reconstruct Sentences"
date: 2024-02-15T04:14:46+01:00
draft: false 
cover:
    image: "anagrams.jpg"
    alt: ''
tags: ['Natural Language Processing', 'Transformers']
Categories: ['NLP','LLM']
---


In this article, I would be fine-tuning OPENAI's GPT2 on a non-trivial task of turning descrambled sentences into their grammatically correct forms using the same words in the sentence. It also provides a walkthrough on how to finetune models in modal's gpu cloud and explores different training, optimization and generation strategies.

Through this project, I aimed to explore the capabilities of large language models in performing text reconstruction - a valuable skill with applications in areas like language learning, content moderation, and semantic search. The article will provide a detailed walkthrough of the finetuning process, including the training strategies, optimization techniques, and generation approaches I experimented with.


![guy from the office](https://images.aicrowd.com/uploads/ckeditor/pictures/457/content_giphy__8_.gif)

This essentially means that we would be teaching GPT2 how to turn sentences from: 
  ```
  The equations expensive. show is optimization computationally that
  ```
 to :
 ```
 The equations show that optimization is computationally expensive.
 ```
 More examples:
 ```
 'the which wiring flow. propose to diagram, method network a reflects signal We visualize',
 'the interaction networks. the gap Finally, analyze chemical the junction between synapse and we',
 'the process The pseudorandom number illustrated in is Mathematica. generator using',
 'in the of structure resulted decrease mutual signal in information. Introducing correlations input-output',
 'statistical estimators functionals. of various of consistent We investigate existence the bounded-memory',
 'rather negative sense. the question This in strong a is in resolved'
 ```
 to 
 ```
'We propose a method to visualize the wiring diagram, which reflects network signal flow.',
 'Finally, we analyze the interaction between the gap junction and the chemical synapse networks.',
 'The process is illustrated using the pseudorandom number generator in Mathematica.',
 'Introducing correlations in signal structure resulted in the decrease of input-output mutual information.',
 'We investigate the existence of bounded-memory consistent estimators of various statistical functionals.',
 'This question is resolved in the negative in a rather strong sense.
 ```



## Data
The dataset consists of 40000 rows of descrambled sentences and their labels, with columns ['text','id','label'].A view object of the dataset:
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
The dataset object is a pyarrow dataset object returned by HuggingFace's Dataset Library.You can find the dataset here on [huggingface](https://huggingface.co/datasets/damilojohn/Text-Descrambling)

## Data Preprocessing

One of the key challenges in finetuning GPT-2 for sentence reconstruction lies in the model's inherent architecture. As a decoder-only model with causal mask attention, GPT-2 is trained to predict the next token based solely on the previous tokens in the sequence. This is fundamentally different from the typical sequence-to-sequence (seq2seq) approach, where the entire input sentence is used as context to predict each token in the target sentence.

In a standard seq2seq task, an encoder-decoder model like the one used in the original Transformer paper is the reasonable choice. The encoder processes the input sentence, and the decoder generates the output sentence by attending to the encoder's representations. This allows the model to directly leverage the input context when producing the target sequence.

However, with a decoder-only model like GPT-2, we need to rethink how we structure the dataset and the training process. Instead of framing this as a seq2seq task, I formatted the dataset as a text completion problem, where the model is shown the entire scrambled sentence and tasked with providing the correctly unscrambled form.

The each row in the dataset was formatted to the following template:

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

**The Preprocess function**

The first three lines of the function creates our prompt template and tokenizes every row in our dataset. Next we calculate the length of the prompt part of the template i.e wrong sentence: {input text} correct sentence: {label}. We then concatenate the input and labels together to form our template and successfully turn our problem to a next word prediction problem.

Then we set every token in the labels before 'correct sentence:' to -101.
Why did I do this?

The reason for doing this is to ensure that the model doesn't waste training steps learning how to predict the precursor part of the sentence and only learns to predict tokens that comes after the term 'correct sentence:'.We want to make sure that we are only backpropagating losses related to descrambling sentences only and not add any noise that might confuse the model further 

This is done by ensuring the model only calculates loss on the actual tokens we need to learn. -101 is an arbitrary token used by the transformers library to tell models to not attend to those token positions, similar to using the attention mask for padding.

Finally the preprocessing is completed by also setting all pad tokens in the labels to -101 for the same exact reason stated above, we don't want our model paying attention to padding tokens.

Here's a visual breakdown of the whole process


## Hardware

For this task, the Nvidia A10G(32GB) provided by [modal](https://modal.com) was used for training, since GPT2-large has 774m parameters and we trained in 16bit precision, meaning we need 774m *16 bytes to store weights, and an additional -- for optimizer states and gradients.Using 16gb VRAM with a reasonable batch size would have been sufficient but training for shorter periods on a supposedly more expensive GPU turned out to be actually cheaper.
```
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
```

## Finetuning

Finetuning in modal was fairly straightforward, all we had to do was create a docker-like image containing all of our needed libraries that runs in a sandboxed [gvisor container](https://cloud.google.com/blog/products/identity-security/open-sourcing-gvisor-a-sandboxed-container-runtime) and installed all our neccessary libraries.

Next we wrote functions such as our trainer class and other utils to be executed in the container environment we created.
For training, I used a learning rate of 3e-5 (5e-5 is typically used to train gpt2 but I tried a smaller learning rate since we had a large batch size).
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
Output: wrong sentence: the which wiring flow. propose to diagram, method network a reflects signal We visualize correct sentence:The diagram, which reflects the wiring network to propose a signal flow.                                                                                                                  
Output: wrong sentence: the interaction networks. the gap Finally, analyze chemical the junction between synapse and we correct sentence:The gap junction and the chemical interaction between the synapse networks.                                                                                                                   
Output: wrong sentence: the process The pseudorandom number illustrated in is Mathematica. generator using correct sentence:The pseudorandom number generator is illustrated using the process in Mathematica.                                                                                                                
Output: wrong sentence: in the of structure resulted decrease mutual signal in information. Introducing correlations input-output correct sentence:This resulted in decrease of mutual correlations in the input-output structure of signal information.                                                                                                               
Output: wrong sentence: statistical estimators functionals. of various of consistent We investigate existence the bounded-memory correct sentence:The existence of bounded-memory estimators consistent with various statistical functionals.                                                                                                                 
Output: wrong sentence: rather negative sense. the question This in strong a is in resolved correct sentence:The question is resolved in a rather strong negative sense.
```

By merely visually inspecting the outputs the first thing I noticed was how easy it was to make up sentences that looked correct but actually make no sense. For example, for the descrambled sentence below:

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

For generation, I used Beam-search as a decoding strategy as it felt like the most reasonable decoding strategy for this task. I experimented with top_p and top_k sampling and the results weren't as good as Beam Search.



In conclusion, the 335m parameter model finetuned on this task performs alot better than the 175b GPT3.5 with few-shot prompting. It was interesting to find that a 335m parameter model could learn to descramble sentences and generalize well on unseen samples

View the code [here](https://github.com/damilojohn/Text-Descrambling)