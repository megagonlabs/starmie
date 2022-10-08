
import pickle
import time
import argparse
import mlflow
import numpy as np
from lsh_search import LSHSearcher
from checkPrecisionRecall import saveDictionaryAsPickleFile, calcMetrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="sato", choices=['sherlock', 'sato', 'cl', 'tapex'])
    parser.add_argument("--benchmark", type=str, default='santos')
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--num_func", type=int, default=16)
    parser.add_argument("--num_table", type=int, default=100)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--scal", type=float, default=1.00)
    # parser.add_argument("--N", type=int, default=10)
    # parser.add_argument("--threshold", type=float, default=0.7)
    # mlflow tag
    parser.add_argument("--mlflow_tag", type=str, default=None)

    hp = parser.parse_args()


    # mlflow logging
    for variable in ["encoder", "num_func", "num_table", "benchmark", "K", "run_id", "scal"]:
        mlflow.log_param(variable, getattr(hp, variable))

    if hp.mlflow_tag:
        mlflow.set_tag("tag", hp.mlflow_tag)


    encoder = hp.encoder
    singleCol = hp.single_column

    dataFolder = hp.benchmark
    # Set augmentation operators, sampling methods, K, and threshold values according to the benchmark
    if 'santos' in dataFolder or dataFolder == 'wdc':
        sampAug = "drop_col_tfidf_entity"
        K = 10
        threshold = 0.7
        if dataFolder == 'santosLarge' or dataFolder == 'wdc':
            K, threshold = hp.K, 0.1
    elif 'tus' in dataFolder:
        sampAug = "drop_cell_alphaHead"
        K = 60
        threshold = 0.1
    singSampAug = "drop_cell_tfidf_entity"

    # If we need to change the value of N, or change the filepath to the pkl files (including indexing), change here:
    #   N: number of returned elements for each query column
    if encoder in ['sherlock', 'sato']:
        N = 50
        query_path = "data/"+dataFolder+"/"+encoder+"_query.pkl"
        table_path = "data/"+dataFolder+"/"+encoder+"_datalake.pkl"
        index_path = "data/"+dataFolder+"/indexes/lsh_"+encoder+".bin"
    else:
        N = 4
        table_id = hp.run_id
        table_path = "data/"+dataFolder+"/vectors/cl_datalake_"+sampAug+"_column_"+str(table_id)+".pkl"
        query_path = "data/"+dataFolder+"/vectors/cl_query_"+sampAug+"_column_"+str(table_id)+".pkl"
        index_path = "data/"+dataFolder+"/indexes/lsh_open_data_"+str(table_id)+"_"+str(hp.scal)+".bin"
        if singleCol:
            N = 50
            table_path = "data/"+dataFolder+"/vectors/cl_datalake_"+singSampAug+"_column_"+str(table_id)+"_singleCol.pkl"
            query_path = "data/"+dataFolder+"/vectors/cl_query_"+singSampAug+"_column_"+str(table_id)+"_singleCol.pkl"
            index_path = "data/"+dataFolder+"/indexes/lsh_open_data_"+str(table_id)+"_singleCol.bin"

    num_hash_func = hp.num_func
    num_hash_table = hp.num_table
    # Call LSHSearcher from lsh_search.py
    searcher = LSHSearcher(table_path, num_hash_func, num_hash_table, hp.scal)
    # Load the query from the pickle file
    queries = pickle.load(open(query_path,"rb"))
    start_time = time.time()
    returnedResults = {}
    avgNumResults = []
    query_times = []

    for q in queries:
        query_start_time = time.time()
        res, numCalls = searcher.topk(encoder,q,K,N=N,threshold=threshold)
        returnedResults[q[0]] = [r[1] for r in res]
        avgNumResults.append(numCalls)
        query_times.append(time.time() - query_start_time)

    print("Average number of verification calls: ", sum(avgNumResults)/len(avgNumResults))
    print("Average QUERY TIME: %s seconds " % (sum(query_times)/len(query_times)))
    print("10th percentile: ", np.percentile(query_times, 10), " 90th percentile: ", np.percentile(query_times, 90))
    print("--- Total Query Time: %s seconds ---" % (time.time() - start_time))

    # santosLarge and WDC benchmarks are used for efficiency
    if hp.benchmark == 'santosLarge' or hp.benchmark == 'wdc':
        print("No groundtruth for %s benchmark" % (hp.benchmark))
    else:
        # Calculating effectiveness scores (Change the paths to where the ground truths are stored)
        if 'santos' in hp.benchmark:
            k_range = 1
            groundTruth = "data/santos/santosUnionBenchmark.pickle"
        else:
            k_range = 10
            if hp.benchmark == 'tus':
                groundTruth = 'data/table-union-search-benchmark/small/tus-groundtruth/tusLabeledtusUnionBenchmark'
            elif hp.benchmark == 'tusLarge':
                groundTruth = 'data/table-union-search-benchmark/large/tus-groundtruth/tusLabeledtusLargeUnionBenchmark'

        calcMetrics(K, k_range, returnedResults, gtPath=groundTruth)
