from .indexer import NMSIndexer
from .datautils import *
import os


class FastKnn(object):
    def __init__(self, data=None, id_dict=None, fastknn_folder=None,
                 index_method="hnsw", index_space='cosinesimil', data_type="dense", dist_type="float", index_M=20,
                 index_efC=200):
        if fastknn_folder:
            ref_id_dict, index, index_params = self.load(fastknn_folder)
            self.ref_id_dict = ref_id_dict
            self.index = index
            self.index_params = index_params
        else:
            ref_id_dict, index, index_params = self.create_fastknn(data, id_dict, index_method, index_space, data_type,
                                                                   dist_type, index_M, index_efC)
            self.ref_id_dict = ref_id_dict
            self.index = index
            self.index_params = index_params

    def load(self, fastknn_folder):
        try:
            ref_id_dict = load_ref_id_dict(fastknn_folder + "/mappings.json")
            index_params = load_dict(fastknn_folder + "/index_params.json")
            index = NMSIndexer(index_path=fastknn_folder + "/index.bin",
                               method=index_params["method"], space=index_params["space"],
                               data_type=index_params["data_type"], dist_type=index_params["dist_type"],
                               M=index_params["M"], efC=index_params["efC"], num_threads=index_params["num_threads"])
            return ref_id_dict, index, index_params
        except:
            raise

    def create_fastknn(self, data, id_dict, index_method, index_space, data_type, dist_type, index_M, index_efC):
        try:
            indexer = NMSIndexer(method=index_method, space=index_space, data_type=data_type, dist_type=dist_type,
                                 M=index_M, efC=index_efC)
            indexer.add_batch_data(data, list(id_dict.keys()))
            indexer.train_index()
            return id_dict, indexer, indexer.index_params
        except:
            raise

    def save(self, fastknn_folder):
        if not os.path.isdir(fastknn_folder):
            os.mkdir(fastknn_folder)
        # Integer index to id mappings
        save_dict(self.ref_id_dict, dict_file_path=fastknn_folder + "/mappings.json")
        # Indexer params
        save_dict(self.index_params, dict_file_path=fastknn_folder + "/index_params.json")
        # Indexer
        self.index.save_index(index_path=fastknn_folder + "/index.bin")

    def query(self, query, k):
        ids, distance = self.index.query_index_batch_by_vector(query, k)
        # Map indexes to ids
        ids = get_mapped_matrix(ids, self.ref_id_dict)
        # Cast distance list to np.array
        distance = np.array([dist for dist in distance])
        return ids, distance

    def query_as_df(self, query, k, query_index=None, nn_column="nearest_neighbours",
                    distance_column="distances", same_ids=False, remove_identity=False):
        if remove_identity:
            # We need to get 1 more nearest neightbours to get k real ones
            k += 1
        ids, distance = self.query(query, k)
        result_df = get_mapped_matrix_as_df(ids, distance, index=query_index, nn_column=nn_column,
                                            distance_column=distance_column)
        cols = list(result_df.columns.values)
        if same_ids:  # then we can map indexes back to real ids
            result_df.loc[:, "id"] = result_df.loc[:, "index"].map(self.ref_id_dict)
            if remove_identity:
                result_df.loc[:, nn_column] = result_df.apply(
                    lambda x: [nn for nn in x[nn_column] if nn != x["id"]][:k - 1],
                    axis=1)
            result_df = result_df[["id"] + cols]
        return result_df
