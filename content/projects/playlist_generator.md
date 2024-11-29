---
title: "Playlist Generator"
date: 2023-08-11T02:14:46+01:00
draft: True
cover:
    image: "vibe_book.png"
    alt: 'Playlist Generator'
tags: ['Semantic Search','Embeddings', 'Transformers']
Categories: ['Machine Learning',Cloud ,'Semantic Search', 'Recsys']
---
# Playlist Generator An app that uses semantic search to generate afrobeat song playlists from input texts  

This is a side project where I built  an end-to-end machine learning application that uses a sentence transformer model hosted as a model-as-a-service. It is a simple recommendation system that tries  to find afrobeat songs about a user's sentiment by comparing the user's text inputs  to song lyrics  and then return the most similar songs. 
![before prompt](https://proglangclassifier.s3.eu-west-2.amazonaws.com/playlist_Gen_space.png)

![after prompt](https://proglangclassifier.s3.eu-west-2.amazonaws.com/playlist_gen_after.png)

## Architecture 
![architecture](https://proglangclassifier.s3.eu-west-2.amazonaws.com/Playlist_Generator+(1).jpeg)

## UI

When a user enters a prompt and clicks on 'get playlist', a POST request containing the user's message is sent to an API Gateway endpoint. The API Gateway triggers the lambda function , our lambda function performs semantic search against a vector database of song lyrics embeddings. The embeddings with the highest cosine similarity score are returned and the songs are returned to the frontend. 

 