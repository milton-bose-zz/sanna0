from __future__ import print_function, division

import numpy as np
import theano
from sklearn.utils.class_weight import compute_sample_weight

def initialize_class_weights(y_data, n_classes=None, name=None,
                             return_numpy_object=True,
                             class_weight=None):

    assert y_data.min() >= 0

    if y_data.ndim == 1:
        class_ = y_data
        if n_classes is None:
            n_classes = len(np.unique(y_data))
        w = np.ones((y_data.shape[0], n_classes))
        w[np.arange(y_data.shape[0]), y_data.astype(int)] = 0
    else:
        class_ = np.argmax(y_data, axis=1)
        if n_classes is not None:
            assert n_classes == y_data.shape[1]
        else:
            n_classes = y_data.shape[1]
        w = 1 - y_data

    assert n_classes >= 2

    w = w / (w.shape[0] * (w.shape[1] - 1))


    sample_weights = compute_sample_weight(class_weight, class_)
    w = w * sample_weights.reshape((-1,1))
    w = w / w.sum()

    if return_numpy_object:
        return w
    else:
        w = theano.shared(w.astype(theano.config.floatX), name=name)
        return w


