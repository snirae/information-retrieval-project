# imports
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import math
from inverted_index_gcp import *

nltk.download('stopwords')


# ranking algorithms
class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index: InvertedIndex, DL: pd.DataFrame, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = DL.shape[0]
        self.AVGDL = DL['length'].mean()
        self.DL = DL  # document length

        self.words, self.pls = zip(*self.index.posting_lists_iter())

    def calc_idf(self, list_of_tokens):
        """
        This function calculates the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        list_of_tokens: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():  # if term is in the index
                n_ti = self.index.df[term]  # number of documents containing term t
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))  # idf formula
            else:
                pass
        return idf

    def score(self, query, N=3):
        """
        This function calculates the bm25 score for every relevant document in the corpus
        and returns the top N documents sorted by score.

        Parameters:
        -----------
        query: list of tokens representing the query. For example: ['look', 'blue', 'sky']
        N: int, number of top documents to retrieve

        Returns:
        -----------
        res: sorted list of (doc_id, score), bm25 scores for top N documents.
        """
        candidates = self.get_candidates(query)
        if len(candidates) == 0:  # if no candidate is found
            return []
        idf = self.calc_idf(query)  # calculate idf for each term in the query
        res = [(can, self._score_pair(query, can, idf)) for can in candidates]  # calculate score for each candidate
        if len(res) < 1:
            return []
        return sorted(res, key=lambda x: x[1], reverse=True)[:N]

    def _score_pair(self, query, doc_id, idf):
        """
        This function calculates the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[self.DL['id'] == doc_id]['length'].values[0]  # document length

        for term in query:
            if term in self.index.df.keys():  # if term is in the index
                term_frequencies = dict(self.pls[self.words.index(term)])  # term frequencies in the corpus
                if doc_id in term_frequencies.keys():  # if term is in the document
                    freq = term_frequencies[doc_id]  # term frequency in the document
                    numerator = idf[term] * freq * (self.k1 + 1)  # numerator of the bm25 formula
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)  # denominator of the bm25 formula
                    score += (numerator / denominator)  # bm25 formula
        return score

    def get_candidates(self, tokens):
        """
        This function returns the set of documents that contain at least one of the terms in the query.

        Parameters:
        -----------
        tokens: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        candidates: list of documents that contain at least one of the terms in the query.
        """
        candidates = []  # list of candidates
        for term in np.unique(tokens):
            if term in self.words:  # if term is in the index
                candidates.append(self.pls[self.words.index(term)])  # add posting list of term to candidates

        if len(candidates) < 1:
            return []
        else:
            candidates = list(reduce(lambda a, b: a + b, candidates))  # flatten list of candidates
            return list(map(lambda x: x[0], candidates))


class CosineSimilarity:
    """
    Cosine similarity calculation
    """
    def __init__(self, index, DL):
        self.index = index
        self.DL = DL  # document length
        self.N = len(DL)

    def score(self, query, N=3):
        """
        This function calculates the cosine similarity with tfidf for every relevant document in the corpus
        and returns the top N documents sorted by score.

        Parameters:
        -----------
        query: list of tokens representing the query. For example: ['look', 'blue', 'sky']
        N: int, number of top documents to retrieve

        Returns:
        -----------
        res: sorted list of (doc_id, score), cosine similarity scores for top N documents.
        """
        candidates_scores = self.get_candidate_documents_and_scores(query)  # get candidates and scores
        Q = self._generate_query_tfidf_vector(query)  # generate query tfidf vector
        D = self._generate_document_tfidf_matrix(candidates_scores)  # generate document tfidf matrix
        sim_dict = self._cosine_similarity(D, Q)  # calculate cosine similarity
        return self._get_top_n(sim_dict, N)  # return top N documents

    def _generate_query_tfidf_vector(self, query_to_search):
        """
        This function generates the tfidf vector for the query.

        Parameters:
        -----------
        query_to_search: list of tokens representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        Q: tfidf vector for the query.
        """
        epsilon = .0000001  # epsilon to avoid division by zero
        total_vocab_size = len(self.index.df)  # total number of terms in the corpus
        Q = np.zeros(total_vocab_size)  # initialize query vector
        term_vector = list(self.index.df.keys())  # list of terms in the corpus
        counter = Counter(query_to_search)  # count number of times each term appears in the query
        for token in np.unique(query_to_search):
            if token in self.index.df.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
                df = self.index.df[token]  # number of documents containing term t
                idf = math.log(self.DL.shape[0] / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)  # index of term in the term vector
                    Q[ind] = tf * idf  # tfidf score
                except:
                    pass
        return Q

    def _generate_document_tfidf_matrix(self, candidates_scores):
        """
        This function generates the tfidf matrix for the documents.

        Parameters:
        ------------
        candidates_scores: list of tuples (doc_id, score) for the relevant documents.

        Returns:
        ------------
        D: tfidf matrix for the documents.
        """
        total_vocab_size = len(self.index.df)  # total number of terms in the index
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])  # unique documents
        D = np.zeros((len(unique_candidates), total_vocab_size))  # initialize tfidf matrix
        D = pd.DataFrame(D)  # convert to dataframe for easier indexing

        D.index = unique_candidates  # set document ids as index
        D.columns = self.index.df.keys()  # set terms as columns

        for key in candidates_scores:  # iterate over all relevant documents
            tfidf = candidates_scores[key]  # get tfidf score
            doc_id, term = key  # get document id and term
            D.loc[doc_id][term] = tfidf  # set tfidf score in the matrix

        return D

    def _cosine_similarity(self, D, Q):
        """
        This function calculates the cosine similarity between the query and the documents.

        Parameters:
        ------------
        D: tfidf matrix for the documents.
        Q: tfidf vector for the query.

        Returns:
        ------------
        sim_dict: dictionary of (doc_id, score) for the documents.
        """
        q_ss = np.sum(Q ** 2)  # sum of squares of query vector
        cosine_func = lambda d: np.dot(d, Q) / np.sqrt(np.sum(d ** 2) * q_ss)  # cosine similarity function
        return {doc_id: cosine_func(d) for doc_id, d in D.iterrows()}  # calculate cosine similarity

    def _get_top_n(self, sim_dict, N=3):
        """
        This function returns the top N documents sorted by score.

        Parameters:
        ------------
        sim_dict: dictionary of (doc_id, score) for the documents.

        Returns:
        ------------
        res: sorted list of (doc_id, score), cosine similarity scores for top N documents.
        """
        return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[:N]  # return top N documents

    def get_candidate_documents_and_scores(self, tokens):
        """
        This function returns the relevant documents and their tfidf scores.

        Parameters:
        ------------
        tokens: list of tokens representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        ------------
        candidates_scores: list of tuples (doc_id, score) for the relevant documents.
        """
        tfidf_scores = {}  # dictionary of (doc_id, score) for the relevant documents
        for token in set(tokens):
            if token in self.index.posting_locs:  # avoid terms that do not appear in the index.
                idf = self.idf(token)  # idf score
                for doc_id, tf in self.index.read_posting_list(token):  # iterate over all documents containing term t
                    tfidf_scores[(doc_id, token)] = tf * idf / self.DL[self.DL['id'] == doc_id]['length'].values[0]  # tfidf score
        return tfidf_scores

    def idf(self, token):
        """
        This function returns the idf score for a term.

        Parameters:
        ------------
        token: term for which idf score is to be calculated.

        Returns:
        ------------
        idf: idf score for the term.
        """
        return np.log(self.N / self.index.df[token])  # idf score


class BinaryRanking:
    """
    This class implements the binary ranking algorithm.
    """
    def __init__(self, index):
        self.index = index  # inverted index

    def score(self, query, N=None):
        """
        This function calculates the binary ranking score for the relevant documents.

        Parameters:
        ------------
        query: list of tokens representing the query. For example: ['look', 'blue', 'sky']
        N: number of documents to return. If None, all relevant documents are returned.

        Returns:
        ------------
        res: sorted list of (doc_id, score) for the relevant documents.
        """
        scores = {}  # dictionary of (doc_id, score) for the relevant documents
        for token in query:
            if token in self.index.df.keys():  # avoid terms that do not appear in the index.
                pl = self.index.read_posting_list(token)  # read posting list for term t
                for (doc_id, tf) in pl:  # iterate over all documents containing term t
                    scores[doc_id] = scores.get(doc_id, 0) + 1  # binary score

        res = sorted([(doc_id, score) for doc_id, score in scores.items()], key=lambda x: x[1], reverse=True)  # sort by score
        return res[:N] if N else res  # return top N documents


############################################################################################################

# class that combines all the algorithms
class SearchAndRank:
    """
    This class contains the search and ranking algorithms.
    """

    english_stopwords = frozenset(stopwords.words('english'))
    wiki_stopwords = ["category", "references", "also", "external", "links",
                      "may", "first", "see", "history", "people", "one", "two",
                      "part", "thumb", "including", "second", "following",
                      "many", "however", "would", "became"]  # corpus specific stopwords
    all_stopwords = english_stopwords.union(wiki_stopwords)

    def __init__(self, title_index=None, doc_title_path=None,
                 t_DL_path=None, body_index=None, b_DL_path=None, anchor_index=None,
                 k1=1.5, b=0.75, pv_path=None, pr_path=None, tokenizer=re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE),
                 stemmer=None, stopwords=all_stopwords):

        self.title_index = title_index
        self.title_DL = pd.read_csv(t_DL_path, index_col=False) if t_DL_path else None

        self.doc_title = pd.read_csv(doc_title_path, index_col=False) if doc_title_path else None

        self.body_index = body_index
        self.body_DL = pd.read_csv(b_DL_path, index_col=False) if b_DL_path else None

        self.anchor_index = anchor_index

        self.cos_sim_body = CosineSimilarity(index=body_index, DL=self.body_DL) if body_index else None
        self.binary_title = BinaryRanking(title_index) if title_index else None
        self.binary_body = BinaryRanking(body_index) if body_index else None
        self.binary_anchor = BinaryRanking(anchor_index) if anchor_index else None

        self.pv = pd.read_csv(pv_path, index_col=False) if pv_path else None
        self.pr = pd.read_csv(pr_path, index_col=False) if pr_path else None

        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopwords = stopwords

    def search(self, query, N=5):
        return self.search_binary_title(query, N)

    # def search_bm25_title(self, query, N=40, tokenize=True):
    #     if tokenize:
    #         query = self.full_tokenize(query)
    #     return self.bm25_title.score(query, N)

    # def search_bm25_body(self, query, N=40, tokenize=True):
    #     if tokenize:
    #         query = self.full_tokenize(query)
    #     return self.bm25_body.score(query, N)

    def search_cos_sim_body(self, query, N=40, tokenize=True):
        """
        This function returns the top N documents for a query using the cosine similarity algorithm and body index.

        Parameters:
        ------------
        query: query string
        N: number of documents to return
        tokenize: whether to tokenize the query or not

        Returns:
        ------------
        res: list of (doc_id, score) for the top N documents
        """
        if tokenize:
            query = self.full_tokenize(query)
        return self.cos_sim_body.score(query, N)

    def search_binary_title(self, query, N=None, tokenize=True):
        """
        This function returns the top N documents for a query using the binary ranking algorithm and title index.

        Parameters:
        ------------
        query: query string
        N: number of documents to return
        tokenize: whether to tokenize the query or not

        Returns:
        ------------
        res: list of (doc_id, score) for the top N documents
        """
        if tokenize:
            query = self.full_tokenize(query)
        return self.binary_title.score(query, N)

    def search_binary_body(self, query, N=None, tokenize=True):
        """
        This function returns the top N documents for a query using the binary ranking algorithm and body index.

        Parameters:
        ------------
        query: query string
        N: number of documents to return
        tokenize: whether to tokenize the query or not

        Returns:
        ------------
        res: list of (doc_id, score) for the top N documents
        """
        if tokenize:
            query = self.full_tokenize(query)
        return self.binary_body.score(query, N)

    def search_binary_anchor(self, query, N=None, tokenize=True):
        """
        This function returns the top N documents for a query using the binary ranking algorithm and anchor index.

        Parameters:
        ------------
        query: query string
        N: number of documents to return
        tokenize: whether to tokenize the query or not

        Returns:
        ------------
        res: list of (doc_id, score) for the top N documents
        """
        if not self.anchor_index:
            return
        if tokenize:
            query = self.full_tokenize(query)
        return self.binary_anchor.score(query, N)

    def get_page_views(self, doc_ids):
        """
        This function returns the page views for a list of document ids.

        Parameters:
        ------------
        doc_ids: list of document ids

        Returns:
        ------------
        res: list of page views for the given document ids
        """
        return list(self.pv[self.pv['id'].isin(doc_ids)]['page_views'])

    def get_page_ranks(self, doc_ids):
        """
        This function returns the page ranks for a list of document ids.

        Parameters:
        ------------
        doc_ids: list of document ids

        Returns:
        ------------
        res: list of page ranks for the given document ids
        """
        return list(self.pr[self.pr['id'].isin(doc_ids)]['page_rank'])

    def tokenize(self, text):
        """
        This function tokenizes a given text using the tokenizer of the search engine.

        Parameters:
        ------------
        text: text to tokenize

        Returns:
        ------------
        res: list of tokens
        """
        return [token.group() for token in self.tokenizer.finditer(text.lower())]

    def filter_tokens(self, tokens):
        """ The function takes a list of tokens, filters out stopwords and
            stem the tokens using `stemmer`.

        Parameters:
        -----------
        tokens: list of tokens

        Returns:
        -----------
        res: list of filtered and stemmed tokens
        """
        if self.stopwords:  # filter out stopwords
            tokens = list(filter(lambda t: t not in self.stopwords, tokens))

        if self.stemmer:  # stem the tokens
            tokens = list(map(lambda t: str(self.stemmer.stem(t)).lower(), tokens))

        return tokens

    def full_tokenize(self, text):
        """
        This function tokenizes and filters a given text.

        Parameters:
        ------------
        text: text to tokenize and filter

        Returns:
        ------------
        res: list of filtered and stemmed tokens
        """
        return self.filter_tokens(self.tokenize(text))

    def get_doc_title(self, doc_id):
        """
        This function returns the title of a document given its id.

        Parameters:
        ------------
        doc_id: id of the document

        Returns:
        ------------
        res: title of the document
        """
        return self.doc_title[self.doc_title['id'] == doc_id]['title'].values[0]

    @staticmethod
    def merge_results(scores1, scores2, weight1=0.5, weight2=0.5, N=3):
        """
        This function merges two lists of (doc_id, score) tuples using the given weights.

        Parameters:
        ------------
        scores1: list of (doc_id, score) tuples
        scores2: list of (doc_id, score) tuples
        weight1: weight of the first list
        weight2: weight of the second list
        N: number of documents to return in the merged list

        Returns:
        ------------
        res: list of (doc_id, score) tuples
        """
        def merge_lists_weighted(l1, l2, w1=0.5, w2=0.5):
            res = []  # list of (doc_id, score) tuples
            i = 0
            j = 0
            l1, l2 = sorted(l1, key=lambda x: x[0]), sorted(l2, key=lambda x: x[0])  # sort the lists by doc_id
            while i < len(l1) and j < len(l2):
                if l1[i][0] == l2[j][0]:  # if the doc_ids are the same, merge the scores
                    res.append((l1[i][0], l1[i][1] * w1 + l2[j][1] * w2))
                    i += 1
                    j += 1
                elif l1[i][0] < l2[j][0]:  # if the doc_id of l1 is smaller, add it to the result
                    res.append((l1[i][0], l1[i][1] * w1))
                    i += 1
                else:  # if the doc_id of l2 is smaller, add it to the result
                    res.append((l2[j][0], l2[j][1] * w2))
                    j += 1
            if i == len(l1):  # if l1 is finished, add the rest of l2 to the result
                res += list(map(lambda x: (x[0], x[1] * w2), l2[j:]))
            if j == len(l2):  # if l2 is finished, add the rest of l1 to the result
                res += list(map(lambda x: (x[0], x[1] * w1), l1[i:]))
            return res

        if not scores1 and not scores2:
            return

        res = merge_lists_weighted(scores1, scores2, weight1, weight2)  # merge the two lists
        res = list(map(lambda x: (int(x[0]), x[1]), np.unique(res, axis=0)))  # remove duplicates
        return sorted(res, key=lambda x: x[1], reverse=True)[:N]  # sort the list by score and return the top N
