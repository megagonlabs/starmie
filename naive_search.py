import numpy as np
import pickle
import random
import heapq
from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm
from bounds import verify, upper_bound_bm, lower_bound_bm, get_edges

class NaiveSearcher(object):
    def __init__(self,
                 table_path,
                 scale,
                 index_path=None
                 ):
        if index_path != None:
            self.index_path = index_path

        # load tables to be queried
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile)
        # For scalability experiments: load a percentage of tables
        self.tables = random.sample(tables, int(scale*len(tables)))
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()

    def topk(self, enc, query, K, threshold=0.6):
        ''' Exact top-k cosine similarity with full bipartite matching
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        if enc == 'sato':
            # For SATO encoder, the first 1187 items in the vector are from Sherlock. The rest are from topic modeling
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in self.tables:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else:
            scores = [(self._verify(query[1], table[1], threshold), table[0]) for table in self.tables]
        scores.sort(reverse=True)
        return scores[:K]

    def topk_bounds(self, enc, query, K, threshold=0.6):
        ''' Algorithm: Pruning with Bounds
            Bounds Techique: reduce # of verification calls
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        H = []
        heapq.heapify(H)
        if enc == 'sato':
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
        satoScore = 0.0
        for table in self.tables:
            # get sherlock and sato components if the encoder is 'sato
            if enc == 'sato':
                tScore = table[1][:, :1187]
                qScore = querySherlock
                sato = table[1][0, 1187:]
                satoScore = self._cosine_sim(querySato, sato)
            else:
                tScore = table[1]
                qScore = query[1]

            # add to heap to get len(H) = K
            if len(H) < K: # len(H) = number of elements in H
                score = verify(qScore, tScore, threshold)
                if enc == 'sato': score = self._combine_sherlock_sato(score, qScore, tScore, satoScore)
                heapq.heappush(H, (score, table[0]))
            else:
                topScore = H[0]
                # Helper method from bounds.py for to reduce the cost of the graph
                edges, nodes1, nodes2 = get_edges(qScore, tScore, threshold)
                lb = lower_bound_bm(edges, nodes1, nodes2)
                ub = upper_bound_bm(edges, nodes1, nodes2)
                if enc == 'sato':
                    lb = self._combine_sherlock_sato(lb, qScore, tScore, satoScore)
                    ub = self._combine_sherlock_sato(ub, qScore, tScore, satoScore)

                if lb > topScore[0]:
                    heapq.heappop(H)
                    score = verify(qScore, tScore, threshold)
                    if enc == 'sato': score = self._combine_sherlock_sato(score, qScore, tScore, satoScore)
                    heapq.heappush(H, (score, table[0]))
                elif ub >= topScore[0]:
                    score = verify(qScore, tScore, threshold)
                    if enc == 'sato': score = self._combine_sherlock_sato(score, qScore, tScore, satoScore)
                    if score > topScore[0]:
                        heapq.heappop(H)
                        heapq.heappush(H, (score, table[0]))
        scores = []
        while len(H) > 0:
            scores.append(heapq.heappop(H))
        scores.sort(reverse=True)
        return scores
        

    def _combine_sherlock_sato(self, score, qScore, tScore, satoScore):
        ''' Helper method for topk_bounds() to calculate sherlock and sato scores, if the encoder is SATO
        '''
        sherlockScore = (1/min(len(qScore), len(tScore))) * score
        full_satoScore = sherlockScore + satoScore
        return full_satoScore

    def topk_greedy(self, enc, query, K, threshold=0.6):
        ''' Greedy algorithm for matching
        Args:
            enc (str): choice of encoder (e.g. 'sato', 'cl', 'sherlock') -- mainly to check if the encoder is 'sato'
            query: the query, where query[0] is the query filename, and query[1] is the set of column vectors
            K (int): choice of K
            threshold (float): similarity threshold. For small SANTOS benchmark, we use threshold=0.7. For the larger benchmarks, threshold=0.1
        Return:
            Tables with top-K scores
        '''
        if enc == 'sato':
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in self.tables:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify_greedy(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else: # encoder is sherlock
            scores = [(self._verify_greedy(query[1], table[1], threshold), table[0]) for table in self.tables]
        scores.sort(reverse=True)
        return scores[:K]

    def _cosine_sim(self, vec1, vec2):
        ''' Get the cosine similarity of two input vectors: vec1 and vec2
        '''
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

    def _verify(self, table1, table2, threshold):
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        graph = np.zeros(shape=(nrow,ncol),dtype=float)
        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                if sim > threshold:
                    graph[i,j] = sim

        max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
        m = Munkres()
        indexes = m.compute(max_graph)
        for row,col in indexes:
            score += graph[row,col]
        return score

    def _verify_greedy(self, table1, table2, threshold):
        nodes1 = set()
        nodes2 = set()
        score = 0.0
        nrow = len(table1)
        ncol = len(table2)
        edges = []
        for i in range(nrow):
            for j in range(ncol):
                sim = self._cosine_sim(table1[i],table2[j])
                if sim > threshold:
                    edges.append((sim,i,j))
                    nodes1.add(i)
                    nodes2.add(j)
        edges.sort(reverse=True)
        for e in edges:
            score += e[0]
            nodes1.discard(e[1])
            nodes2.discard(e[2])
            if len(nodes1) == 0 or len(nodes2) == 0:
                return score
        return score