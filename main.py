# coding: utf-8
import pandas as pd
import os
from indexer import NMSIndexer
from dataprocesser import DataProcesser


def main():
    # Load Data
    df = pd.read_parquet("/Users/fanch/PythonNotebooks/Lab/fast_knn/data/videos.snappy.parquet", engine="fastparquet")

    # Process data
    data_loader = DataProcesser(df,
                                embeddings_column="embedding",
                                id_column="video_id")

    data = data_loader.data
    id_dict = data_loader.id_dict

    ## Create index
    if os.path.exists("indexes/index.bin"):
        data_indexer = NMSIndexer(index_path="indexes/index.bin")
    else:
        data_indexer = NMSIndexer()
        data_indexer.add_batch_data(data, list(id_dict.keys()))
        data_indexer.train_index()
        data_indexer.save_index("indexes/index.bin")

    ## Choose sample vectors
    query = data[:3, :]

    print(query.shape)
    ids, distances = data_indexer.query_index_batch_by_vector(query, k=10)
    print(ids)
    print(distances)

    print(data_loader.get_result_matrix_ids(ids))

    print(data_loader.get_results_as_df(ids))



if __name__ == '__main__':
    main()
