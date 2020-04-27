import numpy as np
import matplotlib.pyplot as plt
# pseudo code for performance metrics

#calculate TP, FP, TN, FN

# using these values, calculate Accuracy, Precision, Recall, F1-measure and consfusion matrix
#######################################
#total positive = total ham files
#total negative = total spam files

#toal files = total negative + total positive

#read test results 
# calculate TP, FP, TN, FN
# also create function that cankeep updating values as each file is processed for testing
# 'test-spam-00400.txt': ['test-spam-00400.txt', 'ham', -4404.628117612352, -4776.3851584504, 'spam', 'wrong']

def calculateMetrics(results):

    TP = 0
    FP = 0 
    TN = 0 
    FN = 0
    for file in results.keys():
        if results.get(file)[1] == results.get(file)[4] :
            if results.get(file)[1] == 'ham':
                TP = TP +1
            else :
                TN = TN  + 1
        elif results.get(file)[1] == 'ham':
            FP = FP +1 
        else :
            FN = FN + 1
            
    accuracy = TP+TN / (TP+FP+FN+TN)
    precision = TP/ (TP+FP)
    recall = TP / (TP+FN)
    F1_Score = 2*(recall * precision) / (recall + precision)
    return TP, FP, TN, FN , accuracy, precision, recall, F1_Score

# accuracy = TP+TN / TP+FP+FN+TN
# precision = TP/ TP+FP
# recall = TP / TP+FN
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)



# using matplotlib
# https://matplotlib.org/3.1.1/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py

# confusion_matrix_data = [[TP, FN],
#                      [FP, TN]]

#columns = ['Ham', 'Spam']
#rows = ['Ham','Spam']

#confusion_matrix = plt.table(cellText = confusion_matrix_data , rowLabels = rows, colLabels = columns)
# plt.title('Confusion Matrix')
# plt.show() 

def plotTable(TP, TN, FP ,FN):
    confusion_matrix_data = [[TP, FN], [FP, TN]]

    columns = ['Ham', 'Spam']
    rows = ['Ham','Spam']
    confusion_matrix = plt.table(cellText = confusion_matrix_data , rowLabels = rows, colLabels = columns)
    plt.title('Confusion Matrix')
    plt.show()