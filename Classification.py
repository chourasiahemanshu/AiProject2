# # Spam Detector. You have to develop a Python-based spam detector using the Na¨ıve Bayes approach.
# # You can only use the following libraries: NumPy, math, re, sys and Matplotlib.
# Your Python program has to be able to build a probabilistic model from the training set (available on
# Moodle). Your code must parse the files in the training set and build a vocabulary with all the words
# it contains. Then, for each word, compute their frequencies and probabilities for each class (class ham
# and class spam).
# To process the texts, fold all characters to lowercase, then tokenize them using the regular expression
# re.split(’\[\^a-zA-Z\]’,aString) and use the set of resulting words as your vocabulary.
# For each word wi
# in the training set, save its frequency and its conditional probability for each class:
# P(wi
# |ham) and P(wi
# |spam). These probabilities must be smoothed using the ‘add δ’ method, with
# δ = 0.5. To avoid arithmetic underflow, work in log10 space.
# Save your model in a text file called model.txt. The format of this file must be the following:
# 1. A line counter i, followed by 2 spaces.
# 2. The word wi
# , followed by 2 spaces.
# 3. The frequency of wi
# in the class ham, followed by 2 spaces.
# 4. The smoothed conditional probability of wi
# in the class ham −P(wi
# |ham), followed by 2 spaces.
# 5. The frequency of wi
# in the class spam, followed by 2 spaces.
# 6. The smoothed conditional probability of wi
# in spam −P(wi
# |spam), followed by a carriage return.
# Note that the file must be sorted alphabetically. For example, your file model.txt could look like the
# following:
# 1 abc 3 0.003 40 0.4
# 2 airplane 3 0.003 40 0.4
# 3 password 40 0.4 50 0.03
# 4 zucchini 0.7 0.003 0 0.000001

import os
import re

trainingDataPath="./projectDetails/train"
testingDataPath ="./projectDetails/test"

def importTrainingFiles():
    listOfFiles = os.listdir("./projectDetails/train")
    print(listOfFiles)
    return listOfFiles


def preprocess_string(str_arg):
    """"
        Preprocessing done includes :
        1. everything apart from letters is excluded
        2. multiple spaces are replaced by single space
        3. str_arg is converted to lower case

    """
    cleaned_str = re.sub('[^a-z\s]+', ' ', str_arg, flags=re.IGNORECASE)  # every char except alphabets is replaced
    cleaned_str = re.sub('(\s+)', ' ', cleaned_str)  # multiple spaces are replaced by single space
    cleaned_str = cleaned_str.lower()  # converting the cleaned string to lower case
    return cleaned_str



def processTheList(fileList):
    for x in fileList:
        f = open(trainingDataPath+"/"+x, "r")
        # f = open(trainingDataPath+"/"+"train-ham-00001.txt", "r")
        aString =  f.read()
        words = preprocess_string(aString)
        listOfWords = re.split(" ", words)
        className = re.split("-",x)[1]
        if className=="ham":
            for word in listOfWords:
                if word in ham_vocab:
                    count = ham_vocab.get(word)
                    count = count+1
                    ham_vocab[word] = count
                else:
                    ham_vocab[word] = 1
        else:
            for word in listOfWords:
                if word in spam_vocab:
                    count = spam_vocab.get(word)
                    count = count + 1
                    spam_vocab[word] = count
                else:
                    spam_vocab[word] = 1

        # print(ham_vocab)
        # print ("words are ",listOfWords)


def calculateTotalWords(spam,ham):
    spamCount = 0
    hamCount = 0
    for a in spam:
        spamCount =  spamCount + spam.get(a)

    for b in ham:
        hamCount = hamCount + ham.get(b)

    return spamCount,hamCount


def calculateConditionalProb(count, vocab):
    cond_prob_vocab = {}
    for a in vocab:
        cond_prob_vocab[a]=vocab.get(a)/count
    return cond_prob_vocab




spam_vocab = {}
ham_vocab = {}

listOfFiles = importTrainingFiles()
processTheList(listOfFiles)
spamCount, hamCount = calculateTotalWords(spam_vocab,ham_vocab)

conditionalSpamProb = calculateConditionalProb(spamCount,spam_vocab)
conditionalHamProb = calculateConditionalProb(hamCount,ham_vocab)


