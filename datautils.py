import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter
import ujson


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


def get_mapped_matrix_as_df(matrix, id_column, id_dict, ref_id_dict, nearest_neighbours_column="nearest_neighbours"):
    result_matrix = get_mapped_matrix(matrix, ref_id_dict)
    df_data = [[id_dict[i], result_matrix[i, :].tolist()] for i in range(len(result_matrix))]

    return pd.DataFrame(df_data, columns=[id_column, nearest_neighbours_column])


def get_data_matrix(df, embeddings_column):
    return np.stack(df.loc[:, embeddings_column].values).astype(np.float32)


def get_id_dict_from_df(df, id_column):
    return defaultdict(lambda: None, df.reset_index().drop("index", axis=1).loc[:, id_column].to_dict())
