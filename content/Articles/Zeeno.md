---
title: "Building A Text based Books Recommendation engine"
date: 2024-05-15T04:14:46+01:00
draft: True
cover:
    image: "zeno-cover.jpg"
    alt: ''
tags: ['Natural Language Processing', 'Transformers']
Categories: ['NLP','LLM']
---

# Building a text-based  books recommendation engine
Find books by describing what they are about.

## Finding books 
What does that really mean? 
A recommendation engine is an algorithm or bunch of algorithms that help users find contents about a particular modality(video, text etc). A text based recommendation engine is one that helps users find information about some text input. 

## Semantic Search 



## Architecture 
[Zeeno](https://zeeno.vercel.app) was built using a combination of semantic search and hybrid keyword search. The backend is built by deploying a serverless AWS lambda function from container images. The lambda function is triggered via API Gateway by queries from the frontend. The AWS Lambda function takes the users query and performs semantic search over a Pinecone vector database.



## Finetuning Embeddings 
The heart of this application is the sentence-transformer that takes user queries and embeds them into vectors that are searched against the embeddings of books in the database. The main idea is to create a latent space where books and queries that describe said books are close to each other. 
To achieve this, I finetuned a sentence-transformer using query-positive pairs and query-negative pairs (using a form of constrastive learning). 

Creating a latent-space that represents user-search intent and books matching those intents was the main aim of finetuning.


## Reranking 
Our recommendation engine could be broke down into two parts :
- The Sentence-Transformer 
- The Reranker 

After the inital step of finding books that are most semantically similar to the user's query. The next step was to use a Reranker to rank those recommendations. 

The sentence-transformer was set to return the N most similar book descriptions to the users query. The Reranker then uses a combination of the user query's vector representation, metadata about the book such as year of release, title, authors etc. to further rank those 30 returned books . The top 10 output of the reranker are then returned to the user.

## Data 
Building a database of books was fun and explored lots of interesting tradeoffs and was an interesting experience in webscraping. 



## CI/CD 
The CI/CD workflow was mainly based on 


## Serverless Hosting and Inference 


## Optimization 



## Putting it all together 

The whole application was built as a fastapi backend. 


## More to come 
Features I plan to integrate:

- the ability to store user queries and returned recommendations.
- 