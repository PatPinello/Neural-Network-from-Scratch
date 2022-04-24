from array import array
from doctest import testmod
import numpy as np
from numpy import append
from nltk import word_tokenize as wt
import json
import NeuralNetwork as NN

#Cleaning Up Text
def RemoveChars(txt):
    removeChars = [',', '(', ')', '.', '@', '?', '!', '/', '#']
    for chars in removeChars:
        txt = txt.replace(chars,'')
    contents = txt
    return contents

#Creating Vocabulary
def CreateVocab(txt):

    #Reading text file and getting the contents
    #engDict = json.load(open("engDict.json"))
    r = open(txt, 'r', encoding="utf-8")
    contents = r.read()

    #Remove bad characters
    contents = RemoveChars(contents)

    #Word Tokenize the cleaned data
    words = wt(contents)

    #Creating dictionary
    vocabRev = {word: i for i, word in enumerate(words)}
     #json.dump(dict, open("dict.json", 'w'))
    return vocabRev

#One-hot Encoding
def OneHotEncode(string, vocabRev):

    #Removing undesired characters
    string = RemoveChars(string)

    #Word Tokenizing and creating dictionary
    stringWords = wt(string)
    #stringVocab = {i: word for i, word in enumerate(stringWords)}

    #List for one-hot encoding
    oneHot = list()

    #Looping through desired words
    #Getting the value in the dictionary for each word
    #Make a row of zeros where the position corresponding to the "word" is 1
    #Each row is a word
    #Append this to the list
    for word in stringWords:
        val = vocabRev[word]
        theWord = [0 for _ in range(len(vocabRev))]
        theWord[val] = 1
        oneHot.append(theWord)

    #Make the list a matrix
    oneHotMat = np.array(oneHot) #input words
    return oneHotMat

#Test Sentence
test = "Bonjour ca va?"
vocabTxt = 'C:\\Users\\patri\\Desktop\\TestTxt.txt'

comment = np.array([[0], [1], [0], [0]]) #One hot for word "comment"
output = OneHotEncode(test, CreateVocab(vocabTxt))

#Created Neural Network
#20 Neurons - Learning Rate of .5
#Columns of input must equal rows of the desired output
nn = NN.NeuralNetwork(output.T, comment, 20, .5)

#Running Neural Network for 2000 epochs
nn.Run(nn, comment, 2000)