"""
Courtesy: https://github.com/cadop/dijkstra
Sparse Version modified by: Lin, Zi Hen
Idea: 
Instead of using shared array for output, 
use a shared dict to store sparse matrix from each process and concatenate.
https://stackoverflow.com/questions/10415028/how-to-get-the-return-value-of-a-function-passed-to-multiprocessing-process

Patches:
1. Change dtype of indices and indptr from int to int32, to match the buffer size
2. Change dtype of shared_prd from "i" to "d", to match the buffer size
"""
import numpy as np
import multiprocessing
from multiprocessing.sharedctypes import RawArray
from scipy.sparse import csgraph, csr_matrix, coo_matrix, vstack
from collections import OrderedDict


def sharedSearch_D(
    indexList, arrPos, sharedCSR, datashape, res_dict, idx
):
    """
    Calculates the shortest path using scipy Dijkstra.  Saves the results to a shared memory array.

    Input

    indexList: used to create a range of indices to calculate shortest path on
        type: [int,int]

    shared_prd: A shared memory Raw array used to store the predecessors
        type: multiprocessing.sharedctypes.RawArray
        length: length of the resulting dataset, given by input arrPos

    shared_dist: A shared memory Raw array used to store the distances
        type: multiprocessing.sharedctypes.RawArray
        length: length of the resulting dataset, given by input arrPos

    sharedCSR: An array of shared memory Raw arrays used to store the parts
    of the CSR matrix and then used to be reconstructed
        type: multiprocessing.sharedctypes.RawArray
        length: 3 arrays, one each for data, indices, indptr

    datashape: The shape of the original data so the csr matrix can be reconstructed
        type: np.shape()

    Returns

    Does not return anything.  Stores all results directly in shared memory dict

    """
    # create array of index list to be calculated
    indexList = range(indexList[0], indexList[1])

    # load the shared csr data and reconstruct it
    data = np.frombuffer(sharedCSR[0], dtype=np.float64)
    indices = np.frombuffer(sharedCSR[1], dtype="int32")
    indptr = np.frombuffer(sharedCSR[2], dtype="int32")
    reconCSR = csr_matrix((data, indices, indptr), shape=(datashape))

    # calculate the shortest path
    distances = csgraph.shortest_path(
        reconCSR, method="D", indices=indexList, directed=True 
    )
    distances[distances == np.inf] = 0
    sparse = coo_matrix(distances)
    res_dict[idx] = sparse 

def multiSearch_sparse(dataset, nprocs):
    """
    Calculates the shortest path using scipy Dijkstra.
    Saves the results to a shared memory array.
    Breaks up a CSR matrix, stores into shared memory, and reconstructs in each process.

    Input

    dataset: data used for shortest path.  in the multiprocessing setup this should be as lightweight as possible
    as it is passed to each process.  Typically a scipy CSR matrix
        type: scipy.csr

    nprocs: number of processors to use in the multiprocessing distribution.  The index lists used
    by dijkstra are split among this number
        type: int

    Returns

    Y_np: The distances from scipy shortest_path
        type: numpy array
        shape: same shape as the dense dataset
    X_np: The predecessors from scipy shortest_path
        type: numpy array
        shape: same shape as the dense dataset

    """
    # create an empty array
    datashape = np.shape(dataset)

    # create shared arrays of the data making up the CSR matrix
    data_array = dataset.data
    indices_array = dataset.indices
    indptr_array = dataset.indptr
    data_size, indices_size, indptr_size = (
        np.size(data_array),
        np.size(indices_array),
        np.size(indptr_array),
    )
    shared_data = RawArray("d", data_size)
    shared_indices = RawArray("i", indices_size)
    shared_indptr = RawArray("i", indptr_size)

    # store the shared array objects in an array to pass to each process
    shared_csr = [shared_data, shared_indices, shared_indptr]
    # put data into the shared data arrays
    shared_data[:] = data_array
    shared_indices[:] = indices_array
    shared_indptr[:] = indptr_array

    # divide work
    chunk_calc = int(datashape[0] / float(nprocs))
    arrStartPos, arrEndPos = 0, 0

    # array to store processes
    procs = []
    
    # create a manager and its dict
    manager = multiprocessing.Manager()
    res_dict = manager.dict()

    for i in range(nprocs):
        # define the section to work on
        startIDX = i * chunk_calc
        endIDX = (i + 1) * chunk_calc
        if i == nprocs - 1:
            endIDX = np.shape(dataset)[0]

        indexList = [startIDX, endIDX]
        # len of flattened data set that will be calculated by search
        datalen = (
            np.shape(dataset[startIDX:endIDX])[0]
            * np.shape(dataset[startIDX:endIDX])[1]
        )

        arrEndPos = arrEndPos + datalen
        arrPos = [arrStartPos, arrEndPos]

        # Start the process
        ptemp = multiprocessing.Process(
            target=sharedSearch_D,
            args=(
                indexList,
                arrPos,
                shared_csr,
                datashape,
                res_dict,
                i
            ),
        )
        ptemp.daemon = True
        ptemp.start()
        procs.append(ptemp)

        # set the position in the flattened array
        arrStartPos += datalen

    # Join the processes back together
    for ptemp in procs:
        ptemp.join()

    # sort manager dict key and loop through dict to concat all the sparse matrix 
    ordered_dict = OrderedDict(sorted(res_dict.items()))
    res = vstack(ordered_dict.values())
    return res


if __name__ == "__main__":
    multiprocessing.freeze_support()
