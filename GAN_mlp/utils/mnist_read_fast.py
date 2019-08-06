def load_mnist(path, is_train=True, one_hot=False):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    if is_train:
        kind = 'train'
    else:
        kind = 't10k'
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    if one_hot:
        labels_ = labels.copy()
        labels = np.zeros((len(labels), 10), dtype=np.int32)
        labels[np.arange(labels_.shape[0]), labels_] = 1

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels
