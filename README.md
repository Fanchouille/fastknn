# FastKnn

## Purpose
Provide a lib to create a fast kNN index and get results as a pandas dataframe.
FastKnn use mainly [nmslib](https://github.com/nmslib/nmslib/) as the kNN backend.


## Anaconda local environment support
Clone project

Install Anaconda local environment as below:
```bash
./install.sh
```

Activate Anaconda local environment as below:

```bash
conda activate ${PWD}/.conda
```

##Use
FastKnn builds a kNN index with specified `index_method` (default : `hnsw`) 
and `index_space` (default : `cosinesimil`)
- See [here](https://github.com/nmslib/nmslib/blob/master/manual/spaces.md) for different spaces
- See [here](https://github.com/nmslib/nmslib/blob/master/manual/methods.md) for different methods

Example :

    # Create index...
    fastknn = FastKnn(data, id_dict)
    
    # Save index
    fastknn.save_fastknn("test_fastknn")
    
    # ...or load if exists
    fastknn = FastKnn(fastknn_folder="test_fastknn")

    # Choose sample vectors
    query = data[:3, :]

    # Query index & get results as df
    results_df = fastknn.query_as_df(query, k=10, same_ids=True, remove_identity=True)

- Where `data` is a m x n numpy array matrix and `id_dict` is a python dictionary with mappings from integer index (0 to m-1) to real ids.

- Once instantiated, `save_fastknn` method saves as files :
    - mappings from integer index to real ids as a json file
    - index parameters as a json file
    - real index object as a bin file
    
- Get a saved FastKnn back by specifying `fastknn_folder`

- Query a FastKnn by using `query_as_df` provided method
    
- `datautils.py` provides method to get `data` and `id_dict` easily from pandas dataframes.


