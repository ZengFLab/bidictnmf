"""
BiDictionaryNMF
================

A PyTorch-based differentiable Non-negative Matrix Factorization model with
*bi-directional dictionaries* (shared + private gene programs), designed for
integrating two batches / datasets while separating shared biology from
batch-specific effects.

Example
-------

>>> from bidictnmf import BiDictionaryNMF
>>> import numpy as np
>>>
>>> model = BiDictionaryNMF(n_shared=20, n_private=10)
>>> model.fit(X, Y)                       # X, Y: (n_cells, n_genes)
>>> H_X_s, H_Y_s = model.get_shared_compositions()
>>> H_new = model.transform(Z, batch='X')
"""

from .model import BiDictionaryNMF

__version__ = "0.2.0"

__all__ = ["BiDictionaryNMF"]
