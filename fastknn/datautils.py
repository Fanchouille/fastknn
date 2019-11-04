import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter
import ujson
from scipy.sparse import csr_matrix


def load_ref_id_dict(dict_file_path):
    with open(dict_file_path, 'r') as fp:
        result = ujson.load(fp)
    result_typed = {int(k): v for (k, v) in result.items()}
    return defaultdict(lambda: None, result_typed)


def load_dict(dict_file_path):
    with open(dict_file_path, 'r') as fp:
        result = ujson.load(fp)
    return defaultdict(lambda: None, result)


def save_dict(id_dict, dict_file_path="mappings/mappings.json"):
    with open(dict_file_path, 'w') as fp:
        ujson.dump(id_dict, fp)


def get_mapped_matrix(matrix, ref_id_dict):
    # Map back integer ids to real ids with id_dict
    result_matrix = np.array([list(itemgetter(*res)(ref_id_dict)) for res in matrix])
    return result_matrix


def get_mapped_matrix_as_df(ids, distances, index=None, nn_column="nearest_neighbours",
                            distance_column="distances"):
    if index is None:
        index = [i for i in range(len(ids))]
    df_data = [[index[i], ids[i, :].tolist(), distances[i, :].tolist()] for i in range(len(ids))]

    result_df = pd.DataFrame(df_data, columns=["index", nn_column, distance_column])
    return result_df


def create_sparse_matrix(df, dim1, dim2, value=None):
    rows, r_pos = np.unique(df.loc[:, dim1], return_inverse=True)
    cols, c_pos = np.unique(df.loc[:, dim2], return_inverse=True)
    if value:
        data = df.loc[:, value].values.astype(np.float32)
    else:
        data = np.ones(r_pos.shape, np.float32)

    id_dict = dict(zip([i for i in range(len(rows))], rows))
    return csr_matrix((data, (r_pos, c_pos))), id_dict


def get_data_matrix(df, embeddings_columns):
    if len(embeddings_columns) == 1:
        return np.stack(df.loc[:, embeddings_columns[0]].values).astype(np.float32)
    else:
        return df.loc[:, embeddings_columns].values.astype(np.float32)


def get_id_dict_from_df(df, id_column):
    return defaultdict(lambda: None, df.reset_index().drop("index", axis=1).loc[:, id_column].to_dict())
