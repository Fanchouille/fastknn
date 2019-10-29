import nmslib


class NMSIndexer(object):
    def __init__(self, method="hnsw", space='cosinesimil', M=20, efC=200, num_threads=4, index_path=None):
        self.method = method
        self.space = space
        self.M = M
        self.efC = efC
        self.num_threads = num_threads
        if index_path:
            self.index = self.load_index(index_path)
        else:
            self.index = self.initialize_index()

    def initialize_index(self):
        index = nmslib.init(method=self.method, space=self.space)
        return index

    def load_index(self, index_path):
        index = self.initialize_index()
        index.loadIndex(index_path, load_data=True)

        return index

    def add_batch_data(self, data, ids):
        self.index.addDataPointBatch(data, ids)

    def train_index(self):
        index_time_params = {'M': self.M, 'indexThreadQty': self.num_threads, 'efConstruction': self.efC, 'post': 2}
        self.index.createIndex(index_time_params)

    def save_index(self, index_path):
        self.index.saveIndex(index_path, save_data=True)

    def query_index_batch_by_vector(self, query, k=10):
        knns = self.index.knnQueryBatch(query, k=k, num_threads=self.num_threads)
        ids = [knn[0] for knn in knns]
        distances = [knn[1] for knn in knns]
        return ids, distances
