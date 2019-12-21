from copy import deepcopy


class Stream(object):

    def fit(self, data, *args, **kwargs):
        for x in data:
            self.update(x, *args, **kwargs)
        return self

    def __call__(self, data, *args, **kwargs):
        self.fit(data, *args, **kwargs)
        return self.value()

    def update(self, x, *args, **kwargs):
        # assert isinstance(x, numbers.Number)
        raise NotImplementedError

    def value(self):
        raise NotImplementedError


class StreamAvg(Stream):
    """
    Computes the average of a stream of data.


    >>> x = np.random.normal(size=10)

    >>> true = np.mean(x)
    >>> est = StreamAcg().fit(x).value()
    >>> assert true == est
    """
    def __init__(self):
        self.n_ = 0
        self.sum_ = 0.0
        self.first = True

    def update(self, x):
        self.n_ += 1
        self.sum_ += x

    def value(self):
        if self.n_ == 0:
            return None

        return self.sum_ / self.n_


class StreamVar(Stream):
    def __init__(self, dof_how='pop'):
        """
        Computes the variance of a stream of N data points using Welford's algorithm
        https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods


        Parameters
        ----------
        dof_how: str, ['pop', 'sample'], numeric
            If pop, rescales by N. If sample, rescales by N-1


        >>> x = np.random.normal(size=10)

        >>> true = np.std(x, ddof=1)**2
        >>> est = StreamVar(dof_how='pop').fit(x).value()
        >>> assert true == est

        >>> true = np.std(x, ddof=0)**2
        >>> est = StreamVar(dof_how='sample').fit(x).value()
        >>> assert true == est

        """
        self.n_ = 0

        self.dof_how = dof_how

        self.A_ = 0
        self.Q_ = 0

        self.first = True

    def __call__(self, data, n_dof='pop'):
        self.n_dof = n_dof
        self.fit(data)
        return self.value()

    def update(self, x):
        self.n_ += 1

        if self.first:
            self.A_ = x
            self.first = False

        else:
            A_prev = deepcopy(self.A_)
            self.A_ = self.A_ + (x - self.A_) / self.n_
            self.Q_ = self.Q_ + (x - A_prev) * (x - self.A_)

    @property
    def dof(self):
        if self.dof_how == 'pop':
            return self.n_ - 1

        elif self.dof_how == 'sample':
            return self.n_

    def value(self):
        return self.Q_ / self.dof
