import numpy as np

def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True):
    n_arrays = len(arrays)
    n_samples = len(arrays[0])
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int(n_samples * (1 - test_size))
    
    result = []
    for array in arrays:
        array = np.array(array)
        train_data = array[indices[:split_idx]]
        test_data = array[indices[split_idx:]]
        result.append(train_data)
        result.append(test_data)
    
    return tuple(result)

