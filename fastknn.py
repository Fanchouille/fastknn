from .indexer import NMSIndexer
from .datautils import *


class FastKnn(object):
    def __init__(self, data, id_dict, fastknn_folder=None,
                 index_method="hnsw", index_space='cosinesimil', index_M=20, index_efC=200):
        if fastknn_folder:
            ref_id_dict, index, index_params = self.load_fastknn()
            self.ref_id_dict = ref_id_dict
            self.index = index
            self.index_params = index_params
        else:
            ref_id_dict, index, index_params = self.create_fastknn(data, id_dict, index_method,
                                                                   index_space, index_M, index_efC)
            self.ref_id_dict = ref_id_dict
            self.index = index
            self.index_params = index_params

    def load_fastknn(self):
        try:
            ref_id_dict = load_ref_id_dict(self.fastknn_folder + "/mappings.json")
            index_params = load_dict(self.fastknn_folder + "/index_params.json")
            index = NMSIndexer(index_path=self.fastknn_folder + "/index.bin")
            return ref_id_dict, index, index_params
        except:
            print("No fastKnn project found at [{}]".format(self.fastknn_folder))

    def create_fastknn(self, data, id_dict, index_method, index_space, index_M, index_efC):
        try:
            indexer = NMSIndexer(method=index_method, space=index_space, M=index_M, efC=index_efC)
            indexer.add_batch_data(data, list(id_dict.keys()))
            indexer.train_index()
            return id_dict, indexer, indexer.index_params
        except:
            print("Bug encountered in create_fastknn")

    def save_fastknn(self, id_dict, indexer, index_params, fastknn_folder):
        # Integer index to id mappings
        save_dict(id_dict, dict_file_path=fastknn_folder + "/mappings.json")
        # Indexer params
        save_dict(index_params, dict_file_path=fastknn_folder + "/index_params.json")
        # Indexer
        indexer.save_index(index_path=fastknn_folder + "/index.bin")

    def query(self, query, k):
        ids, distance = self.index.query_index_batch_by_vector(query, k)
        return
