import os
import re
import math
trainingDataPath="./projectDetails/train"
testingDataPath ="./projectDetails/test"

def importFiles(path):
    listOfFiles = os.listdir(path)
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
    vocab = {}
    for x in fileList:
        f = open(trainingDataPath+"/"+x, "r")
        # f = open(trainingDataPath+"/"+"train-ham-00001.txt", "r")
        aString =  f.read()
        words = preprocess_string(aString)
        listOfWords = re.split(" ", words)
        className = re.split("-",x)[1]
        word_count = 0

        if className == "ham":
            for word in listOfWords:
                word_count = word_count + 1
                if word in vocab:
                    count = vocab.get(word)[0]
                    count = count+1
                    vocab[word][0] = count
                else:
                    vocab[word] = [1,0]
        else:
            for word in listOfWords:
                word_count = word_count + 1
                if word in vocab:
                    count = vocab.get(word)[1]
                    count = count + 1
                    vocab[word][1] = count
                else:
                    vocab[word] = [0,1]

        if className == "ham":
            total_ham_files = total_ham_files+1
            total_ham_word_count = total_ham_word_count + word_count
        else:
            total_spam_files = total_spam_files + 1
            total_spam_word_count = total_spam_word_count + word_count
    # print(vocab)
    return vocab , total_spam_word_count , total_ham_word_count, total_spam_files , total_ham_files



def calculateConditionalProb(ham_count, spam_count,vocab):
    cond_prob_vocab = {}
    vocab_keys = vocab.keys()
    total_unique_words = len(vocab.keys())
    for word in  vocab_keys:
        ham_cond_prob=(vocab.get(word)[0]+0.5)/(ham_count+(total_unique_words*0.5))
        spam_cond_prob=(vocab.get(word)[1]+0.5)/(spam_count+(total_unique_words*0.5))
        cond_prob_vocab[word]=[ham_cond_prob,spam_cond_prob]
    return cond_prob_vocab

#         # Todo: need to verify the formula after smoothing
#         cond_prob_vocab[a]=vocab.get(a)+0.5/count+count*0.5


def testModel(fileList,vocab,ham_class_prob, spam_class_prob):
    testingResult = {}
    for x in fileList:
        f = open(testingDataPath+"/"+x, "r")
        # f = open(trainingDataPath+"/"+"train-ham-00001.txt", "r")
        print(x)
        aString =  f.read()
        words = preprocess_string(aString)
        listOfWords = re.split(" ", words)
        className = re.split("-",x)[1]
        prob_of_beig_ham = math.log(ham_class_prob)
        prob_of_beig_spam = math.log(spam_class_prob)
        for word in listOfWords:
            if(vocab.get(word)):
                prob_of_beig_ham = prob_of_beig_ham + math.log(vocab.get(word)[0])
                prob_of_beig_spam = prob_of_beig_spam + math.log(vocab.get(word)[1])

        if(prob_of_beig_ham > prob_of_beig_spam):
            myclass = "ham"
        else:
            myclass = "spam"

        testingResult[x]= [x,myclass,prob_of_beig_ham,prob_of_beig_spam,className, ("right" if (myclass==className) else "wrong")]

    return testingResult


# Exctracting files
listOfFiles = importFiles(trainingDataPath)
# processing files and cresting the vocabulary

vocabulary , spam_word_count , ham_word_count , spam_file_count , ham_file_count = processTheList(listOfFiles)

# vocabulary is dictionary: with word as key  and value is a [ham_word_count , spam_word_count]
print(vocabulary)
conditionalProb = calculateConditionalProb(ham_word_count,spam_word_count,vocabulary)
classProbOfSpam =  spam_file_count/(spam_file_count+ham_file_count)
classProbOfHam =  ham_file_count/(spam_file_count+ham_file_count)

#todo: showing result in sorted manner

#Todo: testing
# you need to delete 3 files. there is problem of encoding test-spam-65,227,376
testingFileList =  importFiles(testingDataPath)
output = testModel(testingFileList,conditionalProb,classProbOfHam,classProbOfSpam)
print(output)
right = 0
wrong =0
for eachFile in output:
    if( output.get(eachFile)[5]=="right"):
        right+=1
    else:
        wrong+=1

print("Right:",right)
print("wrong:",wrong)




#todo:printing result in the correct manner

#todo: creating the the accuracy, precision, recall and F1-measure for each class (spam and ham), as well as a confusion matrix
