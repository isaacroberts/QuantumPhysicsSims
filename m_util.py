import numpy as np
import numba

def dist(x, y=None):
    if y is not None:
        x = x-y
    return np.linalg.norm(x, axis=-1)

def mag(x):
    # return np.abs(np.sqrt(x.dot(x).sum()))
    return np.linalg.norm(x)

def prob(x):
    return np.abs(x**2).sum() / np.pi

def norm(x):
    magn = np.linalg.norm(x)
    if magn==0:
        return x
    return x / magn

def m_max(x):
    return np.sqrt(np.abs(x.dot(x)).max())

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

def get_slice(arr, axis, start, end):
    if axis==0:
        return arr[start:end]
    elif axis==1:
        return arr[:, start:end]
def set_slice(arr, value, axis, start, end):
    if axis==0:
        arr[start:end] = value
    elif axis==1:
        arr[:, start:end] = value
