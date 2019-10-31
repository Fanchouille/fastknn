# coding: utf-8
import pandas as pd
from fastknn import FastKnn
from fastknn import datautils as du


def main():
    # Load Data
    df = pd.read_parquet("/Users/fanch/PythonNotebooks/Lab/fast_knn/data/videos.snappy.parquet", engine="fastparquet")

    # Process data
    data = du.get_data_matrix(df, "embedding")
    id_dict = du.get_id_dict_from_df(df, "video_id")

    # Create index...
    fastknn = FastKnn(data, id_dict)

    # Save index
    fastknn.save("test_fastknn")

    # ...or load if exists
    fastknn = FastKnn(fastknn_folder="test_fastknn")

    # Choose sample vectors
    query = data[:3, :]

    # Query index & get results as df
    results_df = fastknn.query_as_df(query, k=10, same_ids=True, remove_identity=True)

    print(results_df.loc[:, ["id", "nearest_neighbours"]])


if __name__ == '__main__':
    main()
