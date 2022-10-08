import pickle
import pickle5 as p
import pandas as pd
from matplotlib import *
from matplotlib import pyplot as plt
import numpy as np
import mlflow

def loadDictionaryFromPickleFile(dictionaryPath):
    ''' Load the pickle file as a dictionary
    Args:
        dictionaryPath: path to the pickle file
    Return: dictionary from the pickle file
    '''
    filePointer=open(dictionaryPath, 'rb')
    dictionary = p.load(filePointer)
    filePointer.close()
    return dictionary

def saveDictionaryAsPickleFile(dictionary, dictionaryPath):
    ''' Save dictionary as a pickle file
    Args:
        dictionary to be saved
        dictionaryPath: filepath to which the dictionary will be saved
    '''
    filePointer=open(dictionaryPath, 'wb')
    pickle.dump(dictionary,filePointer, protocol=pickle.HIGHEST_PROTOCOL)
    filePointer.close()


def calcMetrics(max_k, k_range, resultFile, gtPath=None, resPath=None, record=True):
    ''' Calculate and log the performance metrics: MAP, Precision@k, Recall@k
    Args:
        max_k: the maximum K value (e.g. for SANTOS benchmark, max_k = 10. For TUS benchmark, max_k = 60)
        k_range: step size for the K's up to max_k
        gtPath: file path to the groundtruth
        resPath: file path to the raw results from the model
        record (boolean): to log in MLFlow or not
    Return: MAP, P@K, R@K
    '''
    groundtruth = loadDictionaryFromPickleFile(gtPath)
    # resultFile = loadDictionaryFromPickleFile(resPath)
        
    # =============================================================================
    # Precision and recall
    # =============================================================================
    precision_array = []
    recall_array = []
    for k in range(1, max_k+1):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        rec = 0
        ideal_recall = []
        for table in resultFile:
            # t28 tables have less than 60 results. So, skipping them in the analysis.
            if table.split("____",1)[0] != "t_28dc8f7610402ea7": 
                if table in groundtruth:
                    groundtruth_set = set(groundtruth[table])
                    groundtruth_set = {x.split(".")[0] for x in groundtruth_set}
                    result_set = resultFile[table][:k]
                    result_set = [x.split(".")[0] for x in result_set]
                    # find_intersection = true positives
                    find_intersection = set(result_set).intersection(groundtruth_set)
                    tp = len(find_intersection)
                    fp = k - tp
                    fn = len(groundtruth_set) - tp
                    if len(groundtruth_set)>=k: 
                        true_positive += tp
                        false_positive += fp
                        false_negative += fn
                    rec += tp / (tp+fn)
                    ideal_recall.append(k/len(groundtruth[table]))
        precision = true_positive / (true_positive + false_positive)
        recall = rec/len(resultFile)
        precision_array.append(precision)
        recall_array.append(recall)
        if k % 10 == 0:
            print(k, "IDEAL RECALL:", sum(ideal_recall)/len(ideal_recall))
    used_k = [k_range]
    if max_k >k_range:
        for i in range(k_range * 2, max_k+1, k_range):
            used_k.append(i)
    print("--------------------------")
    for k in used_k:
        print("Precision at k = ",k,"=", precision_array[k-1])
        print("Recall at k = ",k,"=", recall_array[k-1])
        print("--------------------------")
    
    map_sum = 0
    for k in range(0, max_k):
        map_sum += precision_array[k]
    mean_avg_pr = map_sum/max_k
    print("The mean average precision is:", mean_avg_pr)

    # logging to mlflow
    if record: # if the user would like to log to MLFlow
        mlflow.log_metric("mean_avg_precision", mean_avg_pr)
        mlflow.log_metric("prec_k", precision_array[max_k-1])
        mlflow.log_metric("recall_k", recall_array[max_k-1])

    return mean_avg_pr, precision_array[max_k-1], recall_array[max_k-1] 