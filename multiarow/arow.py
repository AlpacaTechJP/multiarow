# -*- coding: utf-8 -*-
import numpy

# Adaptive Regularization of Weight Vectors (AROW)
# http://webee.technion.ac.il/people/koby/publications/arow_nips09.pdf


class AROW(object):
    '''Multiclass AROW.

    Attributes
    ----------
    w : numpy array
        Weight vector
    v : numpy array
        Variance vector
    r : float
        Input parameter
    '''

    def __init__(self, r=0.1):
        self.w = {}
        self.v = {}
        self.r = r

    def _update(self, l, x, y, w, v):
        m = numpy.dot(x, w)
        if m * y < 1.0:
            c = numpy.dot(x, x * v)
            b = 1.0 / (c + 1.0 / self.r)
            a = max(0.0, (1.0 - y * m)) * b
            self.w[l] += x * v * (a * y)
            self.v[l] -= v * x * numpy.dot(x, v) * b

    def _compute(self, x, w):
        return numpy.dot(x, w)

    def train(self, label, x):
        '''Learn the new vector with the label.  Label can be new, in which
        case a new class is created.  The label is used as dictionary key
        internally.

        Parameters
        ----------
        label : str
            Class identity.
        x : array-like
            Vector instance under this class label
        '''

        if label not in self.w:
            self.w[label] = numpy.zeros(x.shape)
            self.v[label] = numpy.ones(x.shape)

        for l, w in self.w.items():
            c = self._compute(x, w)
            if l == label:
                y = 1 if c < 0 else 0
            else:
                y = -1 if c >= 0 else 0
            if y != 0:
                self._update(l, x, y, self.w[l], self.v[l])

    def classify(self, x):
        '''Perfom classification based on the current weights
        Parameters
        ----------
        x : array-like
            Vector instance to predict
        Returns
        -------
        label : str
            The most probable class label. Returns None if it's not trained.
        '''
        pred = self.predict(x)
        if pred is not None:
            return pred[0][0]

    def predict(self, x):
        '''Perfom classification based on the current weights
        Parameters
        ----------
        x : array-like
            Vector instance to predict
        Returns
        -------
        scores : [(label0, score0), (label1, score1), ...]
            List of labels with prediction scores.
            Returns None if it's not trained.
        '''
        if len(self.w) == 0:
            return None
        return sorted(
            [(l, self._compute(x, w)) for l, w in self.w.items()],
            key=lambda a: a[1],
            reverse=True)

    def list_label(self):
        '''List up all the current labels.

        Returns
        -------
        labels : list
            List of labels previously trained.
        '''
        return self.w.keys()

    def delete_label(self, label):
        '''Delete label.

        Parameters
        ----------
        label : str
            Class label to delete
        '''
        if label in self.w:
            del self.w[label]
            del self.v[label]
