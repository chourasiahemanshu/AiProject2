import os
import re

trainingDataPath="./projectDetails/train"
testingDataPath ="./projectDetails/test"

def importTrainingFiles():
    listOfFiles = os.listdir("./projectDetails/train")
    # print(listOfFiles)
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


# Todo: Need help with deciding what value counts do we need
# def processTheList(fileList):
#     for x in fileList:
#         f = open(trainingDataPath+"/"+x, "r")
#         # f = open(trainingDataPath+"/"+"train-ham-00001.txt", "r")
#         aString =  f.read()
#         words = preprocess_string(aString)
#         listOfWords = re.split(" ", words)
#         className = re.split("-",x)[1]
#         if className=="ham":
#             for word in listOfWords:
#                 if word in ham_vocab:
#                     count = ham_vocab.get(word)
#                     count = count+1
#                     ham_vocab[word] = count
#                 else:
#                     ham_vocab[word] = 1
#         else:
#             for word in listOfWords:
#                 if word in spam_vocab:
#                     count = spam_vocab.get(word)
#                     count = count + 1
#                     spam_vocab[word] = count
#                 else:
#                     spam_vocab[word] = 1


# Todo: Need help with deciding what value counts do we need

def processTheList(fileList):
    total_spam_word_count = 0
    total_ham_word_count = 0
    total_ham_files =0
    total_spam_files = 0
    for x in fileList:
        f = open(trainingDataPath+"/"+x, "r")
        # f = open(trainingDataPath+"/"+"train-ham-00001.txt", "r")
        aString =  f.read()
        words = preprocess_string(aString)
        listOfWords = re.split(" ", words)
        className = re.split("-",x)[1]
        vocab = {}
        word_count = 0
        for word in listOfWords:
            word_count = word_count+1
            if word in vocab:
                count = vocab.get(word)
                count = count + 1
                vocab[word] = count
            else:
                vocab[word] = 1
        if className == "ham":
            total_ham_files = total_ham_files+1
            total_ham_word_count = total_ham_word_count +word_count
        else:
            total_spam_files = total_spam_files + 1
            total_spam_word_count = total_spam_word_count + word_count

        return vocab , total_spam_word_count , total_ham_word_count, total_spam_files , total_ham_files


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

def calculateSmoothConditionalProb(count, vocab):
    cond_prob_vocab = {}
    for a in vocab:
        # Todo: need to verify the formula after smoothing
        cond_prob_vocab[a]=vocab.get(a)+0.5/count+count*0.5
    return cond_prob_vocab

# def printTrainingData():


# Todo : Can i store my vocab in numpy array
spam_vocab = {}
ham_vocab = {}
# Exctracting files
listOfFiles = importTrainingFiles()
# processing files and cresting the vocabulary
vocabulary , spam_word_count , ham_word_count , spam_file_count , ham_file_count = processTheList(listOfFiles)
# count the total words in each vocab
# spamCount, hamCount = calculateTotalWords(spam_vocab,ham_vocab)

conditionalSpamProb = calculateConditionalProb(spam_word_count,spam_vocab)
conditionalHamProb = calculateConditionalProb(ham_word_count,ham_vocab)

# smoothing
conditionalSpamSmoothProb = calculateSmoothConditionalProb(spam_word_count,vocabulary)
conditionalHamSmoothProb = calculateSmoothConditionalProb(ham_word_count,vocabulary)
#todo: showing result in sorted manner

#Todo: testing

#todo:printing result in the correct manner

#todo: creating the the accuracy, precision, recall and F1-measure for each class (spam and ham), as well as a confusion matrix
