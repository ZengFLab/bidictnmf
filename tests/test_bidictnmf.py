import os
import numpy as np
import pytest

from bidictnmf import BiDictionaryNMF


@pytest.fixture(scope="module")
def data():
    np.random.seed(0)
    n_genes, n_shared, n_private = 200, 6, 3
    n_X, n_Y = 60, 50

    W_true = np.abs(np.random.randn(n_genes, n_shared))
    W_true /= np.linalg.norm(W_true, axis=0, keepdims=True)
    Wx = np.abs(np.random.randn(n_genes, n_private)) * 0.5
    Wy = np.abs(np.random.randn(n_genes, n_private)) * 0.5

    Hx = np.abs(np.random.randn(n_shared + n_private, n_X))
    Hy = np.abs(np.random.randn(n_shared + n_private, n_Y))

    X = (W_true @ Hx[:n_shared] + Wx @ Hx[n_shared:]
         + 0.1 * np.random.randn(n_genes, n_X)).T
    Y = (W_true @ Hy[:n_shared] + Wy @ Hy[n_shared:]
         + 0.1 * np.random.randn(n_genes, n_Y)).T
    return X, Y, n_shared, n_private, n_X, n_Y


def test_fit_runs(data):
    X, Y, n_shared, n_private, *_ = data
    model = BiDictionaryNMF(n_shared=n_shared, n_private=n_private,
                            max_iter=20, verbose=False)
    model.fit(X, Y)
    assert model._fitted
    assert len(model.loss_history) > 0
    # loss should be finite and generally decreasing
    assert np.isfinite(model.loss_history[-1])


def test_shared_compositions_shape(data):
    X, Y, n_shared, n_private, n_X, n_Y = data
    model = BiDictionaryNMF(n_shared=n_shared, n_private=n_private,
                            max_iter=20, verbose=False)
    model.fit(X, Y)
    HX, HY = model.get_shared_compositions()
    assert HX.shape == (n_shared, n_X)
    assert HY.shape == (n_shared, n_Y)


def test_transform(data):
    X, Y, n_shared, n_private, n_X, n_Y = data
    model = BiDictionaryNMF(n_shared=n_shared, n_private=n_private,
                            max_iter=20, verbose=False)
    model.fit(X, Y)
    Z = X[:10]
    H = model.transform(Z, batch='X')
    assert H.shape[0] == n_shared + n_private
    assert H.shape[1] == Z.shape[0]
    assert np.all(H >= 0)


def test_save_load_roundtrip(data, tmp_path):
    X, Y, n_shared, n_private, n_X, n_Y = data
    path = str(tmp_path / "model.pt")

    m1 = BiDictionaryNMF(n_shared=n_shared, n_private=n_private,
                         max_iter=20, verbose=False)
    m1.fit(X, Y)
    W1 = m1.W_shared_np.copy()
    HX1, HY1 = m1.get_shared_compositions()

    m1.save(path)
    assert os.path.exists(path)

    m2 = BiDictionaryNMF.load(path)
    assert m2._fitted
    assert m2.n_shared == n_shared and m2.n_private == n_private
    np.testing.assert_allclose(m2.W_shared_np, W1, atol=1e-6)
    HX2, HY2 = m2.get_shared_compositions()
    np.testing.assert_allclose(HX2, HX1, atol=1e-6)
    np.testing.assert_allclose(HY2, HY1, atol=1e-6)

    # transform still works after reload
    H = m2.transform(X[:5], batch='X')
    assert H.shape == (n_shared + n_private, 5)


def test_properties(data):
    X, Y, n_shared, n_private, *_ = data
    model = BiDictionaryNMF(n_shared=n_shared, n_private=n_private,
                            max_iter=10, verbose=False)
    model.fit(X, Y)
    for attr in ("W_shared_np", "W_X_np", "W_Y_np", "H_X_np", "H_Y_np"):
        assert getattr(model, attr).ndim >= 1
