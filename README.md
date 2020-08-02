# GenerativeTextUsing GRUs and RNNs
This is a deep learning project where I use a Gated Recurrent Network for Generative text writing to auto generate state of the union like text.

## Libraries Used
1) Keras
2) Tensorflow
3) NLTK
4) Re
5) os
6) Pickle

## Input
The input data is a part of the NLTK text data and contains various state of the union documents from as far back as 1940. The aim of this project was to create a DL model which has the ability to auto fill a predicted character after a sequence of 40 characters and do this iteratively 400 times, each time moving forward by 1 character from the start and generating the next character.

## Output
It is clear that over lower thresholds the model produces better text and the more epochs it trains over, the better it gets at predicting the next character.

## Shortcomings
This model can be seen to be quite slow in training as the dataset for training is about 600000 input sequences long. 

# Improvements
A larger dropout layer can be used in between stacked GRUs and RNNs, A different weight initializer and regularizer may also be used to offset the overfitting of the model. Also the model can be trained over more Epochs to improve learning
