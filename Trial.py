from BayesClassifier import *
from DataReader import *

def testClassifier(outputLabel):
    bc = BayesClassifier()
    bc.train(outputLabel + ".train")
    reader = DataReader(outputLabel + ".test")
    correctLabel = {}
    numberGuess = {}
    correct = 0.0
    total = 0.0
    for label, tokens in reader:
        if not label in correctLabel:
            correctLabel[label] = 0.0
        guess = bc.classify(" ".join(tokens))
        if not guess in numberGuess:
            numberGuess[guess] = 0.0
        if guess == label:
            correctLabel[guess] += 1
            correct += 1
        numberGuess[guess] += 1
        total += 1
    for label in correctLabel:
        print "Correct " + label, "-", correctLabel[label]/numberGuess[label]
    print "Total accuracy -", correct / total
        
dataTopic = "movies"
split("data/movies.data", dataTopic)
testClassifier(dataTopic)
