import pandas as pd


def retain_pandas(X, f):
    if type(X) == pd.DataFrame:
        return pd.DataFrame(f(X), index=X.index, columns=X.columns)
    else:
        return f(X)


def get_mismatches(a, b):
    A = set(a)
    B = set(b)

    in_a_not_b = A.difference(B)
    in_b_not_a = B.difference(A)
    return in_a_not_b, in_b_not_a
