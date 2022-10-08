
import numpy as np
import random
import os

from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm


def cosine_sim(vec1, vec2):
    assert vec1.ndim == vec2.ndim
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))


def verify(table1, table2, threshold=0.6):
    score = 0.0
    nrow = len(table1)
    ncol = len(table2)
    graph = np.zeros(shape=(nrow,ncol),dtype=float)
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i],table2[j])
            if sim > threshold:
                graph[i,j] = sim
    max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
    m = Munkres()
    indexes = m.compute(max_graph)
    for row,col in indexes:
        score += graph[row,col]
    return score

def upper_bound_bm(edges, nodes1, nodes2):
    '''
        Calculate the upper bound of the bipartite matching
        Input:
        table1/table2: two tables each of which is with a set of column vectors
        threshold: the minimum cosine similarity to include an edge 
        Output:
        The upper bound of the bipartite matching score (no smaller than true score)
    '''
    score = 0.0
    for e in edges:
        score += e[0]
        nodes1.discard(e[1])
        nodes2.discard(e[2])
        if len(nodes1) == 0 or len(nodes2) == 0:
            return score
    return score

def lower_bound_bm(edges, nodes1, nodes2):
    '''
    Output the lower bound of the bipartite matching score (no larger than true score)
    '''
    score = 0.0
    for e in edges:
        if e[1] in nodes1 and e[2] in nodes2:
            score += e[0]
            nodes1.discard(e[1])
            nodes2.discard(e[2])
        if len(nodes1) == 0 or len(nodes2) == 0:
            return score
    return score


def get_edges(table1, table2, threshold):
    '''
    Generate the similarity graph used by lower bounds and upper bounds
    Args:
        table1 (numpy array): the vectors of the query (# rows: # columns in a table, #cols: dimension of embedding)
        table2 (numpy array): similar to table1, set of column vectors of the data lake table
        threshold (float): minimum cosine similarity to include an edge
    Return:
        list of edges and sets of nodes used in lower and upper bounds calculations
    '''
    nrow = len(table1)
    ncol = len(table2)
    edges = []
    nodes1 = set()
    nodes2 = set()
    for i in range(nrow):
        for j in range(ncol):
            sim = cosine_sim(table1[i],table2[j])
            if sim > threshold:
                edges.append((sim,i,j))
                nodes1.add(i)
                nodes2.add(j)
    edges.sort(reverse=True)
    return edges, nodes1, nodes2