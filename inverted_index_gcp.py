import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from collections import defaultdict
from contextlib import closing
from functools import reduce

# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 1999998


class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket. 
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self._f.close()
                self.upload_to_gcp()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def upload_to_gcp(self):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        file_name = self._f.name
        blob = self.bucket.blob(f"postings_gcp/{file_name}")
        blob.upload_from_filename(file_name)


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, base_dir, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f'{base_dir}/{f_name}', 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


from collections import defaultdict
from contextlib import closing

TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally), 
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of 
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are 
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents 
        # the number of bytes from the beginning of the file where the posting list
        # starts. 
        self.posting_locs = defaultdict(list)

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage 
            side-effects).
        """
        if not tokens:
            return

        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file: 
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary. 
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields 
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs, self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    def set_base_dir(self, base_dir):
        """ Set the base directory for the index. """
        self.base_dir = base_dir

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            idx = pickle.load(f)
        idx.set_base_dir(base_dir)
        return idx

    def read_posting_list(self, w):
        with closing(MultiFileReader()) as reader:
            locs = self.posting_locs[w]
            b = reader.read(base_dir=self.base_dir, locs=locs, n_bytes=self.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter("..", bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp()
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name)
        return bucket_id

    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(f"postings_gcp/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")

    @staticmethod
    def merge_lists(lists):
        lists = [tuple(l) for l in lists]
        idx = {l: 0 for l in lists}
        res = []

        while len(lists) > 0:
            t = reduce(lambda a, b: a if a[idx[a]] < b[idx[b]] else b, lists)
            res.append(t[idx[t]])
            idx[t] += 1
            lists = list(filter(lambda l: idx[l] < len(l), lists))

        return res

    def merge_indices(self, base_dir, names, output_name):
        """ A function that merges the (partial) indices built from subsets of
            documents, and writes out the merged posting lists.
        Parameters:
        -----------
            base_dir: str
                Directory where partial indices reside.
            names: list of str
                A list of index names to merge.
            output_name: str
                The name of the merged index.
        """
        indices = [InvertedIndex.read_index(base_dir, name) for name in names]
        iters = [idx.posting_lists_iter() for idx in indices]
        self.posting_locs = defaultdict(list)
        #### POSTINGS: merge & write out ####

        pls = {}  # term: [(iter_idx, posting_list), ...]
        idxs = list(range(len(iters)))
        writer = MultiFileWriter(base_dir, output_name)
        while True:
            for i in idxs:
                term, pl = next(iters[i], (None, None))
                if pl is not None:
                    pls[term] = pls.get(term, []) + [(i, pl)]

            if not bool(pls):
                break

            curr = min(pls.keys())
            idxs = list(map(lambda x: x[0], pls[curr]))

            merged_pl = InvertedIndex.merge_lists(list(map(lambda x: x[1], pls[curr])))

            self._posting_list = {curr: merged_pl}
            self.df.update({curr: len(merged_pl)})
            self.term_total.update({curr: sum([i[1] for i in merged_pl])})

            for id, tf in merged_pl:
                self.DL[id] = self.DL.get(id, 0) + tf

            self.write_a_posting_list(curr, writer)
            pls.pop(curr)

        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, output_name)
