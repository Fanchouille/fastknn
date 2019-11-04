# FastKnn

## Purpose
Provide a lib to create a fast kNN index and get results as a pandas dataframe.
FastKnn use mainly [nmslib](https://github.com/nmslib/nmslib/) as (fast) kNN backend.


## Install
`pip install git+https://github.com/Fanchouille/fastknn.git`

## Use
FastKnn builds a kNN index with specified `index_method` (default: `hnsw`) 
and `index_space` (default: `cosinesimil`)
- See [here](https://github.com/nmslib/nmslib/blob/master/manual/spaces.md) for different spaces
- See [here](https://github.com/nmslib/nmslib/blob/master/manual/methods.md) for different methods

This code has been tested with `hnsw` method and `cosinesimil` / `l2` space for dense data and `cosinesimil_sparse` / `cosinesimil_sparse_fast` space.

Example with dense data :
    
    from fastknn import FastKnn
    
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

- Where `data` is a m x n numpy array matrix and `id_dict` is a python dictionary with mappings from integer index (0 to m-1) to real ids.
    - `fastknn.datautils` provides method to get `data` and `id_dict` easily from pandas dataframes.

- Once instantiated, `save` method saves as files :
    - mappings from integer index to real ids as a json file
    - index parameters as a json file
    - index as a bin file
    
- Get a saved FastKnn back by specifying `fastknn_folder`

- Query a FastKnn by using `query_as_df` provided method with the following parameters 
    - `query` - p x n numpy array - matrix to be matched to `data`
    - `k` - integer - the number of nearest neighbours
    - `nn_column` - string - name of resulting column containing the nearest neighbours (default: `nearest_neighbours`)
    - `distance_column` - string - name of resulting column containing the distances to nearest neighbours (default: `distances`)
    - `same_ids` - bool - when querying the same data that was indexed, gets index + real ids (default: `False`)
    - `remove_identity` - bool - when querying the same data that was indexed, get `k` nearest neighbours without the perfect identity match (default: `False`)


## Development
Clone project

Install Anaconda local environment as below:
```bash
./install.sh
```

Activate Anaconda local environment as below:

```bash
conda activate ${PWD}/.conda
```
