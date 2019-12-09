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
        print(results_df.loc[:, ["id", "nearest_neighbours"]])

    elif data_type_demo == "TEXT":
        # TEXT DATA
        print("*" * 100)
        print("TEXT DATA")
        print("*" * 100)

        # Load recipe dataset
        df = pd.read_json("data/recipe_dataset.json")
        df.loc[:, "ingredients"] = df.loc[:, "ingredients"].map(lambda x: " ".join(ing for ing in x))
        df = df.head(1000)

        # Process data
        data, id_dict = df.loc[:, "ingredients"].to_numpy(), du.get_id_dict_from_df(df, id_column="id")

        # Create index...
        # index_space = "leven" for levenstein distance => use dist_type = "int"
        # /!\ data is expected to be a list
        fastknn = FastKnn(data=data.tolist(), id_dict=id_dict, index_space="leven", data_type="string", dist_type="int")

        # Save index
        fastknn.save("test_fastknn_text")

        # ...or load if exists
        fastknn = FastKnn(fastknn_folder="test_fastknn_text")

        # Choose sample vectors (pass query as a list here)
        query_index = [0, 1, 2]
        query = data[query_index].tolist()

        # Query index & get results as df
        results_df = fastknn.query_as_df(query, k=10, query_index=query_index, same_ids=True, remove_identity=True)
        print(results_df.loc[:, ["id", "nearest_neighbours"]])

    else:
        pass


if __name__ == '__main__':
    main("TEXT")
