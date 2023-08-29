---
title: " Image Localization and Object Distance Prediction with Pytorch."
date: 2023-08-16T04:14:46+01:00
draft: true
cover:
    image: "iss.jpg"
    alt: 'image sensor'
tags: ['Computer Vision','Deep Learning', 'Pytorch']
Categories: ['Machine Learning','Computer Vision','Deep Learning']
---

#
Docking to the ISS would have been more fun if we actually had to dock with the international space station, regardless, we have an interesting problem of trying to create an automatic ISS docker by estimating the coordinates and distance of the international space station from images. 


In this article , we would be using deep learning to estimate the coordinates of an object in an image and also training the model to predict the distance of images . While there are other non-Machine learning approaches to the problem, the aim of this article is too see how predicting the distance of an image from a camera can be framed as a regression problem and that hopefully our model learns how to predict the distance of an object from the camera .
<image> 

We would be using the NASA ISS docking dataset(because space is fun) which consists of images of the International Space Station and a csv file containing the image distances and the coordinates of their location . Hence this is an image localization and regression problem . 
To start with, lets take a look at our dataset . 
We have a folder consisting of 10,000 train images,5000 test images and 1000 images for training validation. 

The following code snippet visualizes the coordinates of a random image of the ISS from the training set and displays the distance from the space shuttle together with an image

<image>

```
index = np.random.randint(0,train_df.shape[0]-1)
#reading the image
image_no = str(train_df['ImageID'][index])
slash = '/'
img = Image.open(r'C:\Users\USER\Datasets\ISS docking\Train\train' + slash + str(image_no) + '.jpg')
distance = train_df['distance'][index]
location = eval(train_df['location'][index])
#draw img
draw = ImageDraw.Draw(img)
x,y,r = location[0],location[1],3
points = (x-r,y-r,x+r,y+r)
draw.ellipse(points,'green')
draw.text((0,0),f'Distance:{distance}','yellow')
img 
```
The dataset class provided by torchvision is used to store all our images