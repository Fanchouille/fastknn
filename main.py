# coding: utf-8
import pandas as pd
from fastknn import FastKnn
from fastknn import datautils as du


def main(data_type_demo="DENSE"):
    if data_type_demo == "DENSE":
        # DENSE DATA
        print("*" * 100)
        print("DENSE DATA")
        print("*" * 100)

        # Load Data
        df = pd.read_csv("data/data_banknote_authentication.txt", delimiter=",", header=None)

        # Process data
        data = du.get_data_matrix(df, [0, 1, 2, 3])
        # Fake index here (identity mapping)
        id_dict = dict(zip(df.index.values.tolist(), df.index.values.tolist()))
        # Get target (last col) - binary class 0/1
        target = du.get_id_dict_from_df(df, 4)

        # Create index...
        fastknn = FastKnn(data=data, id_dict=id_dict, target=target, index_space="l2")

        # Save index
        fastknn.save("test_fastknn_dense")

        # ...or load if exists
        fastknn = FastKnn(fastknn_folder="test_fastknn_dense")

        # Choose sample vectors
        query_index = [0, 1, 2]
        query = data[query_index, :]

        # Query index & get results as df
        results_df = fastknn.query_as_df(query, k=10, query_index=query_index, same_ids=True, remove_identity=True)
        print(results_df)

        # Predict with target metadata
        prediction_df = fastknn.prediction_as_df(query, k=10, query_index=query_index,
                                                 same_ids=True, remove_identity=True, prediction_type="classification")
        print(prediction_df)

    elif data_type_demo == "SPARSE":
        # SPARSE DATA
        print("*" * 100)
        print("SPARSE DATA")
        print("*" * 100)

        # Load movielens dataset
        df = pd.read_csv("data/movielens.csv")

        # Process data
        data, id_dict = du.create_sparse_matrix(df, dim1="movieName", dim2="userId", value="rating")

        # Create index...
        fastknn = FastKnn(data=data, id_dict=id_dict, index_space="cosinesimil_sparse", data_type="sparse")

        # Save index
        fastknn.save("test_fastknn_sparse")

        # ...or load if exists
        fastknn = FastKnn(fastknn_folder="test_fastknn_sparse")

        # Choose sample vectors
        query_index = [563, 837, 8603]
        query = data[query_index, :]

        # Query index & get results as df
        results_df = fastknn.query_as_df(query, k=10, query_index=query_index, same_ids=True, remove_identity=True)

    else:
        pass


if __name__ == '__main__':
    main("DENSE")
