# Lex Gospodinoff
# March 23, 2013
# BayesClassifier.py
# This class uses labeled data to train a naive Bayes classifier.

import math, os, pickle
from DataReader import *

LAMBDA = 0.005

class BayesClassifier:
   
   def __init__(self):
      '''Initializes the classifier.'''
      self.counts = {}
      self.labels = {}
      self.totalDocs = 0.0
      self.allSeen = set()
            
   def train(self, dataFile):   
      '''Trains the Naive Bayes Sentiment Classifier.'''
      reader = DataReader(dataFile)
      for label, tokens in reader:
         self.totalDocs += 1
         if label not in self.counts:
            self.labels[label] = {'docCount':1.0, 'wordCount':0.0}
            self.counts[label] = {}
         else:
            self.labels[label]['docCount'] += 1
         for word in tokens:
            if word.lower() not in self.counts[label]:
               self.counts[label][word.lower()] = 1.0
               self.allSeen.add(word.lower())
            else:
               self.counts[label][word.lower()] += 1
            self.labels[label]['wordCount'] += 1
      
   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive or negative ).
      '''
      target = tokenize(sText)
      best = -(10**15) # something that will be smaller than anything
      bestLabel = 'warning: mistake' # a dummy; this should never come up
      for label in self.counts:
         prob = 0
         for word in target:
            if word in self.counts[label]:
               prob += math.log((self.counts[label][word] + LAMBDA) / (self.labels[label]['wordCount'] + len(self.allSeen) * LAMBDA))
            elif word in self.allSeen:
               prob += math.log(LAMBDA / (self.labels[label]['wordCount'] + len(self.allSeen) * LAMBDA))
         if prob > best:
            best = prob
            bestLabel = label
      return bestLabel

   def save(self, sFilename):
      '''Save the learned data during training to a file using pickle.'''

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(self.counts)
      p.dump(self.labels)      
      p.dump(self.totalDocs)
      p.dump(self.allSeen)     
      f.close()
   
   def load(self, sFilename):
      '''Given a file name of stored data, load and return the object stored in the file.'''

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      self.counts = u.load()
      self.labels = u.load()
      self.totalDocs = u.load()
      self.allSeen = u.load()
      f.close()
