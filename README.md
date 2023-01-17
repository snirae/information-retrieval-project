# information-retrieval-project
BGU Data Engineering third year 1st semester information retrieval course project - English wikipedia search engine

-------------------------------------------------------------------------------------------------------------------

`inverted_index_gcp.py`:

This code defines a set of classes for building and manipulating an inverted index,
which is a data structure commonly used in information retrieval and text analysis.

The InvertedIndex class is the main class, which is used to build the index from a set of documents,
where each document is represented as a list of tokens.

The MultiFileWriter and MultiFileReader classes are used to write and read the index from disk,
in a way that allows for large indexes that don't fit in memory.

The InvertedIndex class uses the MultiFileWriter to write the posting lists to disk,
and the MultiFileReader to read them back from disk when querying the index.
Additionally, it uses a 'google cloud storage' bucket to save the posting files.
