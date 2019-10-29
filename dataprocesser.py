import pandas as pd
import numpy as np
from collections import defaultdict
from operator import itemgetter


class DataProcesser(object):
    def __init__(self, df, embeddings_column="embeddings", id_column="id"):
        self.df = df.reset_index().drop("index", axis=1)
        self.id_column = id_column
        self.embeddings_column = embeddings_column
        self.id_dict = self.get_id_dict(self.df, self.id_column)
        self.data = self.get_data_matrix(self.df, self.embeddings_column)

    def get_data_matrix(self, df, embeddings_column):
        return np.stack(df.loc[:, embeddings_column].values).astype(np.float32)

    def get_id_dict(self, df, id_column):
        return defaultdict(lambda: None, df.loc[:, id_column].to_dict())

    def get_result_matrix_ids(self, matrix):
        # Map back integer ids to real ids with id_dict
        result_matrix = np.array([list(itemgetter(*res)(self.id_dict)) for res in matrix])
        return result_matrix

    def get_results_as_df(self, matrix, nearest_neighbours_column="nearest_neighbours"):
        result_matrix = self.get_result_matrix_ids(matrix)
        df_data = [[self.id_dict[i], result_matrix[i, :].tolist()] for i in range(result_matrix.shape[0])]

        return pd.DataFrame(df_data, columns=[self.id_column, nearest_neighbours_column])
