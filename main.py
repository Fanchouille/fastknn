# coding: utf-8
import pandas as pd
import os
from indexer import NMSIndexer
from datautils import DataUtils


def main():
    # Load Data
    df = pd.read_parquet("/Users/fanch/PythonNotebooks/Lab/fast_knn/data/videos.snappy.parquet", engine="fastparquet")

    # Process data
    data_utils = DataUtils(df,
                           embeddings_column="embedding",
                           id_column="video_id",
                           ref_mappings_path="mappings/mappings.json")

    # Create index or load if exists
    if os.path.exists("indexes/index.bin"):
        data_indexer = NMSIndexer(index_path="indexes/index.bin")
    else:
        data_indexer = NMSIndexer()
        data_indexer.add_batch_data(data_utils.data, list(data_utils.ref_id_dict.keys()))
        data_indexer.train_index()
        data_indexer.save_index("indexes/index.bin")

    # Choose sample vectors
    query = data_utils.data

    ids, distances = data_indexer.query_index_batch_by_vector(query, k=10)

    print(data_utils.get_mapped_matrix_as_df(ids, nearest_neighbours_column="nearest_neighbours"))


if __name__ == '__main__':
    main()
