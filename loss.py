# coding: utf-8

import scipy as sp

def logloss(act, pred):
    """
    log loss
    :param act:
    :param pred:
    :return:
    """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(
        sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    print '-------------logloss-----------', ll
    return ll