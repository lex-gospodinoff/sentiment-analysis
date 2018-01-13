# sentiment-analysis
Naive Bayes Classifier, written in Python for CSCI 311 at Middlebury College
This program uses Bayes classification to sort text documents by label. In the training data, documents are 
analyzed word-by-word to determine the correlation between each word and each of the possible labels. The 
program is then able to evaluate new documents and determine their most probable label. 

The four datasets are:
20news: This is a dataset of messages from news boards, labeled by which of 20 different topics they belong to.
auto_aviation: Messages relating to either cars or planes, labeled as either "auto" or "aviation".
movies: Reviews of movies, labeled as either positive or negative (this is where the project gets its name).
real_sim: This dataset contains messages relating to real cars, labeled "real", and messages relating to a 
racing video game, labeled "sim".
