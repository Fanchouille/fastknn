import nmslib

data_types = {"dense": nmslib.DataType.DENSE_VECTOR,
              "string": nmslib.DataType.OBJECT_AS_STRING,
              "sparse": nmslib.DataType.SPARSE_VECTOR}

dist_types = {"float": nmslib.DistType.FLOAT,
              "int": nmslib.DistType.INT}


class NMSIndexer(object):
    def __init__(self, method="hnsw", space='cosinesimil', data_type="dense", dist_type="float", M=20, efC=500,
                 num_threads=4, index_path=None, efS=500):
        self.index_params = {"method": method,
                             "space": space,
                             "data_type": data_type,
                             "dist_type": dist_type,
                             "M": M,
                             "efC": efC,
                             "efS": efS,
                             "num_threads": num_threads}
        if index_path:
            self.index = self.load_index(index_path, self.index_params)
        else:
            self.index = self.initialize_index(self.index_params)

    def initialize_index(self, index_params):
        data_type = data_types[index_params["data_type"]]
        dist_type = dist_types[index_params["dist_type"]]
        index = nmslib.init(method=index_params["method"], space=index_params["space"], data_type=data_type,
                            dtype=dist_type)
        return index

    def load_index(self, index_path, index_params):
        index = self.initialize_index(index_params)
        if (index_params["method"] == "hnsw") and (index_params["space"] in ["cosinesimil", "l2"]):
            index.loadIndex(index_path, load_data=False)
        else:
            index.loadIndex(index_path, load_data=True)

        return index

    def add_batch_data(self, data, ids):
        self.index.addDataPointBatch(data, ids)

    def train_index(self):
        index_time_params = {'M': self.index_params["M"], 'indexThreadQty': self.index_params["num_threads"],
                             'efConstruction': self.index_params["efC"], 'post': 2}
        self.index.createIndex(index_time_params)

    def save_index(self, index_path):
        if (self.index_params["method"] == "hnsw") and (self.index_params["space"] in ["cosinesimil", "l2"]):
            self.index.saveIndex(index_path, save_data=False)
        else:
            self.index.saveIndex(index_path, save_data=True)

    def query_index_batch_by_vector(self, query, k=10):
        self.index.setQueryTimeParams({'efSearch': self.index_params["efS"]})
        knns = self.index.knnQueryBatch(query, k=k, num_threads=self.index_params["num_threads"])
        print(len(knns))
        ids = [knn[0] for knn in knns]
        distances = [knn[1] for knn in knns]
        return ids, distances
