from array import array
from doctest import testmod
import numpy as np
from numpy import append
from nltk import word_tokenize as wt
import json
import NeuralNetwork as NN

#Creating Vocabulary
vocabTxt = 'C:\\Users\\patri\\Desktop\\TestTxt.txt'
removeChars = [',', '(', ')', '.', '@', '?', '!', '/', '#']

#engDict = json.load(open("engDict.json"))
r = open(vocabTxt, 'r', encoding="utf-8")
contents = r.read()

for chars in removeChars:
    contents = contents.replace(chars,'')
words = wt(contents)

vocab = {i: word for i, word in enumerate(words)}
vocabRev = {word: i for i, word in enumerate(words)}
vocabMatrix = np.zeros((len(vocab),len(vocab)))
#json.dump(dict, open("dict.json", 'w'))

for key in vocab:
	vocabMatrix[key,key] = 1

#Test Sentence
test = "Bonjour ca va?"
for chars in removeChars:
    test = test.replace(chars,'')
testWords = wt(test)
testVocab = {i: word for i, word in enumerate(testWords)}
oneHot = list()

for word in testWords:
    val = vocabRev[word]
    theWord = [0 for _ in range(len(vocab))]
    theWord[val] = 1
    oneHot.append(theWord)
oneHotMat = np.array(oneHot)
y = np.array([[0, 1, 0, 0]])
nn = NN.NeuralNetwork(oneHotMat, y, 20, .5)
nn.Run(nn, y, 1)