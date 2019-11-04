# coding: utf-8
import pandas as pd
from fastknn import FastKnn
from fastknn import datautils as du
from fastknn.indexer import NMSIndexer
import numpy as np

def main(data_type_demo="DENSE"):
    if data_type_demo == "DENSE":
        # DENSE DATA
        print("*" * 100)
        print("DENSE DATA")
        print("*" * 100)

        # Load Data
        #df = pd.read_parquet("/Users/fanch/PythonNotebooks/Lab/fast_knn/data/videos.snappy.parquet",
        #                     engine="fastparquet")
        #df = df.head(100)
        df = pd.read_csv("/Users/fanch/Downloads/thumbnails_exploded.csv", delimiter=";")


        # Process data
        data = du.get_data_matrix(df,[str(x) for x in np.arange(1280).tolist()])
        id_dict = du.get_id_dict_from_df(df, "Unnamed: 0")

        # Create index...
        fastknn = FastKnn(data, id_dict, index_space="cosinesimil")

        # Save index
        fastknn.save("test_fastknn_dense")

        # ...or load if exists
        fastknn = FastKnn(fastknn_folder="test_fastknn_dense")

        # Choose sample vectors
        query = data[:3, :]

        # Query index & get results as df
        results_df = fastknn.query_as_df(query, k=10, same_ids=True, remove_identity=True)
        print(results_df.loc[:, ["id", "nearest_neighbours"]])

    elif data_type_demo == "SPARSE":
        # SPARSE DATA
        print("*" * 100)
        print("SPARSE DATA")
        print("*" * 100)

        # Load movielens dataset
        df = pd.read_csv("movielens/movielens.csv")

        # Process data
        data, id_dict = du.create_sparse_matrix(df, dim1="movieName", dim2="userId", value="rating")

        # Create index...
        fastknn = FastKnn(data=data, id_dict=id_dict, index_space="cosinesimil_sparse", data_type="sparse")

        # Save index
        fastknn.save("test_fastknn_sparse")

        # ...or load if exists
        fastknn = FastKnn(fastknn_folder="test_fastknn_sparse")

        # Choose sample vectors
        query = data[[563, 837, 8603], :]

        # Query index & get results as df
        results_df = fastknn.query_as_df(query, k=10, query_index=[563, 837, 8603], same_ids=True, remove_identity=True)
        print(results_df.loc[:, ["id", "nearest_neighbours"]])

    else:
        pass


if __name__ == '__main__':
    main("DENSE")
