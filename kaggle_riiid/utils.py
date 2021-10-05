import numpy as np
import pickle 

from collections.abc import MutableMapping
import random

from numba import njit

def read_from_pickle(filename):
    with open(filename, 'rb') as handle:
        d = pickle.load(handle)
    return d

# use protocol 4 since Kaggle is on Python 3.7 and does not support protocol 5 yet
def write_to_pickle(d, filename, protocol=4):
    with open(filename, 'wb') as handle:
        pickle.dump(d, handle, protocol=protocol)

def read_from_pickle_dict_chunks(filename):
    d = {}
    with open(filename, "rb") as handle:
        while True:
            try:
                d_chunk = pickle.load(handle)
            except EOFError:
                break
            d.update(d_chunk)
    return d

def write_to_pickle_dict_chunks(d, filename, chunk_size=10, protocol=4):
    def make_chunks(d, chunk_size):
        chunk = {}
        size = 0
        for k, v in d.items():
            chunk[k] = v
            size += 1
            if size == chunk_size:
                yield chunk
                size = 0
                chunk.clear()
        if chunk:
            yield chunk

    with open(filename, 'wb') as handle:
        for chunk in make_chunks(d, chunk_size):
            pickle.dump(chunk, handle, protocol=protocol)
        
def pipe(first, *args):
    for fn in args:
        first = fn(first)
    return first

@njit
def mahalanobis(x, y, vinv=np.eye(8, dtype=np.float64)):
    result = 0.0

    diff = np.empty(x.shape[0], dtype=np.float64)

    for i in range(x.shape[0]):
        diff[i] = x[i] - y[i]

    for i in range(x.shape[0]):
        tmp = 0.0
        for j in range(x.shape[0]):
            tmp += vinv[i, j] * diff[j]
        result += tmp * diff[i]

    return np.sqrt(result)

@njit
def pdist_sim(arr, i, j, n):
    if i == j:
        return 1
    if i < j:
        i, j = j, i
    return arr[n*j - j*(j+1)//2 + i - 1 - j]

@njit 
def np_array(*args):
    return np.array(args)

@njit
def np_nans(size):
    return np.full(size, np.nan)

@njit
def np_sum(arr):
    sum = 0
    for x in arr:
        if not np.isnan(x):
            sum += x
    return sum

@njit
def np_absmean(arr):
    sum = 0
    count = 0
    for x in arr:
        if not np.isnan(x):
            sum += abs(x)
            count += 1
    return sum/count if count > 0 else np.nan

@njit
def np_mean(arr):
    sum = 0
    count = 0
    for x in arr:
        if not np.isnan(x):
            sum += x
            count += 1
    return sum/count if count > 0 else np.nan

@njit
def np_std(arr):
    sum = 0
    count = 0
    for x in arr:
        if not np.isnan(x):
            sum += x
            count += 1

    if count == 0: 
        return np.nan
    
    mean = sum/count
    sum2 = 0
    for x in arr:
        if not np.isnan(x):
            sum2 += (x-mean)**2 
    return np.sqrt(sum2/count)

@njit 
def np_max(arr):
    max_v = -np.inf
    for v in arr:
        if v > max_v:
            max_v = v
    return max_v

@njit 
def np_min(arr):
    min_v = np.inf
    for v in arr:
        if v < min_v:
            min_v = v
    return min_v

@njit
def np_slope(x_, y_):
    x = x_[~(np.isnan(x_) | np.isnan(y_))]
    y = y_[~(np.isnan(x_) | np.isnan(y_))]
    x_diff = x-np.mean(x)
    y_diff = y-np.mean(y)
    sum_x_diff_sq = np.sum(x_diff**2)
    return np.sum(x_diff * y_diff)/sum_x_diff_sq if sum_x_diff_sq != 0 else np.nan

class arraylist:
    def __init__(self, space=100):
        self.space = space
        self.data = np_nans(self.space)
        self.capacity = self.space
        self.size = 0

    def update(self, row):
        for r in row:
            self.append(r)

    def append(self, x):
        if self.size == self.capacity:
            self.capacity += self.space
            newdata = np_nans(self.capacity)
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

class arraylist2D:
    def __init__(self, space=(100, 2)):
        self.space = space
        self.data = np_nans(space)
        self.capacity = space
        self.size = 0

    def update(self, row):
        for r in row:
            self.append(r)

    def append(self, x):
        if self.size == self.capacity[0]:
            self.capacity = (self.capacity[0] + self.space[0], self.capacity[1])
            newdata = np_nans(self.capacity)
            newdata[:self.size, :] = self.data
            self.data = newdata

        self.data[self.size, :] = x
        self.size += 1

class RandomDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        """ Create RandomDict object with contents specified by arguments.
        Any argument
        :param *args:       dictionaries whose contents get added to this dict
        :param **kwargs:    key, value pairs will be added to this dict
        """
        # mapping of keys to array positions
        self.keys = {}
        self.values = []
        self.last_index = -1

        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        if key in self.keys:
            i = self.keys[key]
            self.values[i] = (key, val)
        else:
            self.last_index += 1
            i = self.last_index
            self.values.append((key, val))

        self.keys[key] = i
    
    def __delitem__(self, key):
        if key not in self.keys:
            raise KeyError

        # index of item to delete is i
        i = self.keys[key]
        # last item in values array is
        move_key, move_val = self.values.pop()

        if i != self.last_index:
            # we move the last item into its location
            self.values[i] = (move_key, move_val)
            self.keys[move_key] = i
        # else it was the last item and we just throw
        # it away

        # shorten array of values
        self.last_index -= 1
        # remove deleted key
        del self.keys[key]
    
    def __getitem__(self, key):
        if key not in self.keys:
            raise KeyError

        i = self.keys[key]
        return self.values[i][1]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return self.last_index + 1

    def random_key(self):
        """ Return a random key from this dictionary in O(1) time """
        if len(self) == 0:
            raise KeyError("RandomDict is empty")
        
        i = random.randint(0, self.last_index)
        return self.values[i][0]

    def random_value(self):
        """ Return a random value from this dictionary in O(1) time """
        return self[self.random_key()]

    def random_item(self):
        """ Return a random key-value pair from this dictionary in O(1) time """
        k = self.random_key()
        return k, self[k]