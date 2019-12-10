from .indexer import NMSIndexer
from .datautils import *
import os


class FastKnn(object):
    def __init__(self, data=None, id_dict=None, target=None, fastknn_folder=None,
                 index_method="hnsw", index_space='cosinesimil', data_type="dense", dist_type="float", index_M=20,
                 index_efC=200):
        if fastknn_folder:
            ref_id_dict, index, index_params, target = self.load(fastknn_folder)
            self.ref_id_dict = ref_id_dict
            self.index = index
            self.index_params = index_params
            self.target = target
        else:
            ref_id_dict, index, index_params, target = self.create_fastknn(data, id_dict, target, index_method,
                                                                           index_space,
                                                                           data_type, dist_type, index_M, index_efC)
            self.ref_id_dict = ref_id_dict
            self.index = index
            self.index_params = index_params
            self.target = target

    def load(self, fastknn_folder):
        try:
            ref_id_dict = load_ref_id_dict(fastknn_folder + "/mappings.json")
            index_params = load_dict(fastknn_folder + "/index_params.json")
            index = NMSIndexer(index_path=fastknn_folder + "/index.bin",
                               method=index_params["method"], space=index_params["space"],
                               data_type=index_params["data_type"], dist_type=index_params["dist_type"],
                               M=index_params["M"], efC=index_params["efC"], num_threads=index_params["num_threads"])
            if os.path.exists(fastknn_folder + "/target.json"):
                target = load_ref_id_dict(fastknn_folder + "/target.json")
            else:
                target = None
            return ref_id_dict, index, index_params, target
        except:
            raise

    def create_fastknn(self, data, id_dict, target, index_method, index_space,
                       data_type, dist_type, index_M, index_efC):
        try:
            indexer = NMSIndexer(method=index_method, space=index_space, data_type=data_type, dist_type=dist_type,
                                 M=index_M, efC=index_efC)
            indexer.add_batch_data(data, list(id_dict.keys()))
            indexer.train_index()
            return id_dict, indexer, indexer.index_params, target
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
        # Target
        if self.target:
            save_dict(self.target, dict_file_path=fastknn_folder + "/target.json")

    def query(self, query, k):
        ids, distance = self.index.query_index_batch_by_vector(query, k)
        # Map indexes to ids
        ids = get_mapped_matrix(ids, self.ref_id_dict)
        # Cast distance list to np.array
        distance = np.array([dist for dist in distance])
        return ids, distance

    def query_as_df(self, query, k=10, query_index=None, nn_column="nearest_neighbours",
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
                result_df.loc[:, distance_column] = result_df.apply(
                    lambda x: [x[distance_column][i] for i in range(len(x[nn_column]))
                               if x[nn_column][i] != x["id"]][: k - 1],
                    axis=1)
                result_df.loc[:, nn_column] = result_df.apply(
                    lambda x: [nn for nn in x[nn_column] if nn != x["id"]][:k - 1],
                    axis=1)
            result_df = result_df[["id"] + cols]

        return result_df

    def most_frequent_value(self, class_list):
        (values, counts) = np.unique(class_list, return_counts=True)
        class_pred = values[counts.argmax()]
        proba = counts.max() / len(class_list)
        return class_pred, proba

    def prediction_as_df(self, query, k=10, query_index=None, same_ids=False, remove_identity=False,
                         prediction_type="classification"):
        result_df = self.query_as_df(query, k=k, query_index=query_index, same_ids=same_ids,
                                     remove_identity=remove_identity)

        if same_ids:  # then we can map indexes back to real ids
            ref_index_dict = {v: k for k, v in self.ref_id_dict.items()}
            result_df.loc[:, "nn_targets"] = result_df.loc[:, "nearest_neighbours"] \
                .map(lambda x: [ref_index_dict[i] for i in x]).map(lambda x: [self.target[i] for i in x])
        else:
            result_df.loc[:, "nn_targets"] = result_df.loc[:, "nearest_neighbours"] \
                .map(lambda x: [self.target[i] for i in x])

        if prediction_type == "classification":
            result_df.loc[:, "predictions"] = result_df.loc[:, "nn_targets"].map(self.most_frequent_value)

        elif prediction_type == "regression":
            result_df.loc[:, "predictions"] = result_df.loc[:, "nn_targets"].map(np.mean)

        result_df = result_df.drop("nn_targets", axis=1)

        return result_df
