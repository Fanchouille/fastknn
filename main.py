# coding: utf-8
import pandas as pd
from fastknn import FastKnn
import datautils as du


def main():
    # Load Data
    df = pd.read_parquet("/Users/fanch/PythonNotebooks/Lab/fast_knn/data/videos.snappy.parquet", engine="fastparquet")

    # Process data
    data = du.get_data_matrix(df, "embedding")
    id_dict =  du.get_id_dict_from_df(df, "video_id")

    # Create index or load if exists
    fastknn = FastKnn(data, id_dict)
    fastknn.save_fastknn("test_fastknn")

    # Choose sample vectors
    query = data[:3,:]

    results_df = fastknn.query_as_df(query, k=10)

    print(results_df)


if __name__ == '__main__':
    main()
