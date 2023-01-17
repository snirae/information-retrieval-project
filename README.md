# information-retrieval-project
BGU Data Engineering third year 1st semester information retrieval course project - English wikipedia search engine

-------------------------------------------------------------------------------------------------------------------

## Classes

in `inverted_index_gcp.py`

### InvertedIndex
The InvertedIndex class is used to build an inverted index from a set of documents,
where each document is represented as a list of tokens.
The index is stored on disk using the MultiFileWriter class, and can be queried using the MultiFileReader class.

### MultiFileWriter and MultiFileReader
The MultiFileWriter and MultiFileReader classes are used to write and read the index from disk,
in a way that allows for large indexes that don't fit in memory.
The InvertedIndex class uses the MultiFileWriter to write the posting lists to disk,
and the MultiFileReader to read them back from disk when querying the index.

in `search_utils.py`

### BM25
The BM25 class is used to rank the results of a query using the BM25 ranking algorithm. 
The BM25 algorithm takes into account the term frequency in the query and the document,
as well as the inverse document frequency of the term. It's the most accurate, but also very slow on large
corpuses.

### CosineSimilarity
The CosineSimilarity class is used to rank the results of a query using the cosine similarity algorithm on tf-idf scores.
The cosine similarity algorithm measures the similarity between two vectors by taking the dot product of the vectors and
dividing by the product of the magnitudes of the vectors.

### BinaryRanking
The BinaryRanking class is used to rank the results of a query using the binary ranking algorithm.
The binary ranking algorithm simply returns all documents that contain at least one of the query terms.
That's the most naive algorithm, but also the fastest one.

### SearchAndRank
The SearchAndRank class is used to search a corpus of documents using an inverted index
and rank the results using one of the ranking algorithms above.
The class takes as input the path to the index and to other relevant files (depending on the level of algorithms support you desire).
The results are returned as a list of tuples, where each tuple contains the document ID and the score.


## The search engine - `search_frontend.py`
The code defines a Flask web application that provides a search function for the corpus.
The MyFlaskApp class is a subclass of the Flask class, and is used to initialize and run the web application.
The run method of the MyFlaskApp class is called when the application is started, and it loads the necessary data,
reads the inverted indices and initializes a SearchAndRank object.

The search and search_body routes are defined, each corresponding to a different endpoint that the user can access via an HTTP request.
The search route takes a query parameter from the request and uses the SearchAndRank object to search the corpus and returns
up to 100 of the best search results for the query. 
The search_body route also takes a query parameter, but this time it uses the inverted index of the body of the articles only and returns the results using TFIDF and Cosine Similarity.
