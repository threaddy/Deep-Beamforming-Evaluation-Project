import numpy as np


class DataGenerator(object):
    def __init__(self, batch_size, gtype, te_max_iter=None):
        assert gtype in ['train', 'test']
        self._batch_size_ = batch_size
        self._gtype_ = gtype
        self._te_max_iter_ = te_max_iter

    def generate(self, xs, ys):
        x = xs[0]
        y = ys[0]
        batch_size = self._batch_size_
        n_samples = len(y)

        index = np.arange(n_samples)
        np.random.shuffle(index)

        iter = 0
        epoch = 0
        pointer = 0
        while True:
            if (self._gtype_ == 'test') and (self._te_max_iter_ is not None):
                if iter == self._te_max_iter_:
                    break
            iter += 1
            if pointer >= n_samples:
                epoch += 1
                if self._gtype_ == 'test' and (epoch == 1):
                    break
                pointer = 0
                np.random.shuffle(index)

            batch_idx = index[pointer: min(pointer + batch_size, n_samples)]
            pointer += batch_size
            yield x[batch_idx], y[batch_idx]