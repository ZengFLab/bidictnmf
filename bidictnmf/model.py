import os
import warnings
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None
    _HAS_TQDM = False

warnings.filterwarnings('ignore')


class BiDictionaryNMF(torch.nn.Module):
    """
    双字典 NMF（PyTorch 可微版本）

    接口与 sklearn 风格保持一致：
        model = BiDictionaryNMF(n_shared=20, n_private=10)
        model.fit(X, Y)
        H_X_s, H_Y_s = model.get_shared_compositions()
        H_new = model.transform(Z, batch='X')
        model.save('model.pt')
        model2 = BiDictionaryNMF.load('model.pt')

    Parameters
    ----------
    n_shared : int
        共享 GP 数量，必须 >= 1。
    n_private : int
        私有 GP 数量，允许为 0（退化为单一共享字典，等价于普通 NMF）。
    lam_shared : float
        W_shared 的 L1 正则（鼓励稀疏可解释）
    lam_private : float
        W_X, W_Y 的 L1 正则（把批次效应压进私有字典）。
        当 n_private=0 时该项恒为 0，不会报错。
    lr : float
        Adam 学习率
    max_iter : int
        最大迭代
    tol : float
        早停阈值
    device : str
        'cuda' or 'cpu'
    verbose : bool
        是否显示 tqdm 进度条。False 则静默训练
    """

    # 需要随模型一起保存/加载的配置项
    _CONFIG_KEYS = (
        'n_shared', 'n_private', 'lam_shared', 'lam_private',
        'lr', 'max_iter', 'tol', 'device',
    )

    def __init__(self, n_shared=20, n_private=10,
                 lam_shared=0.0, lam_private=0.1,
                 lr=1e-3, max_iter=500, tol=1e-5,
                 device=None, verbose=True):
        super().__init__()

        if n_shared < 1:
            raise ValueError(f"n_shared must be >= 1, got {n_shared}.")
        if n_private < 0:
            raise ValueError(f"n_private must be >= 0, got {n_private}.")

        self.n_shared = n_shared
        self.n_private = n_private
        self.n_total = n_shared + n_private
        self.lam_shared = lam_shared
        self.lam_private = lam_private
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.loss_history = []
        self._fitted = False

    # ==================== 内部工具 ====================
    def _has_private(self):
        """是否有私有字典。n_private=0 时为 False，此时退化为单一共享字典。"""
        return self.n_private > 0

    def _build_W(self, batch):
        """
        按 batch 组装完整字典 W。

        - n_private > 0：返回 torch.cat([W_shared, W_X 或 W_Y], dim=1)
        - n_private = 0：直接返回 W_shared（避免对形状 (g, 0) 的空张量做 cat）

        返回形状：(n_genes, n_total)
        """
        if not self._has_private():
            return self.W_shared
        if batch == 'X':
            return torch.cat([self.W_shared, self.W_X], dim=1)
        elif batch == 'Y':
            return torch.cat([self.W_shared, self.W_Y], dim=1)
        else:
            raise ValueError("batch must be 'X' or 'Y'.")

    def _private_norm(self):
        """私有字典 L1 范数；n_private=0 时返回 0。"""
        if not self._has_private():
            return torch.tensor(0.0, device=self.device)
        return torch.sum(torch.abs(self.W_X)) + torch.sum(torch.abs(self.W_Y))

    # ==================== 参数初始化 ====================
    def _init_params(self, X, Y):
        g = X.shape[1]

        # ---- 共享字典 W_shared ----
        XY = torch.cat([X, Y], dim=0).cpu().numpy()
        pca = PCA(n_components=min(self.n_shared, XY.shape[0], XY.shape[1]))
        pca.fit(XY)
        W_s_np = np.abs(pca.components_.T[:, :self.n_shared])
        if W_s_np.shape[1] < self.n_shared:
            extra = np.abs(np.random.randn(g, self.n_shared - W_s_np.shape[1])) * 0.01
            W_s_np = np.hstack([W_s_np, extra])

        self.W_shared = torch.nn.Parameter(
            torch.tensor(W_s_np, dtype=torch.float32, device=self.device))

        # ---- 私有字典（n_private=0 时为形状 (g, 0) 的空张量）----
        # scale 用 max(n_private, 1) 兜底，避免 n_private=0 时除以 0。
        scale = float(torch.mean(X).item()) / max(self.n_private, 1) * 0.1
        self.W_X = torch.nn.Parameter(
            torch.abs(torch.randn(g, self.n_private, device=self.device)) * scale)
        self.W_Y = torch.nn.Parameter(
            torch.abs(torch.randn(g, self.n_private, device=self.device)) * scale)

        # ---- H 初始化 ----
        # 通过 _build_W 组装完整字典，n_private=0 时即为 W_shared 本身。
        with torch.no_grad():
            Wx = self._build_W('X')  # (g, n_total)
            Wy = self._build_W('Y')

            # pinv @ X.T -> (n_total, n)
            H_X_init = torch.relu(torch.linalg.pinv(Wx) @ X.T)
            H_Y_init = torch.relu(torch.linalg.pinv(Wy) @ Y.T)

        self.H_X = torch.nn.Parameter(H_X_init)
        self.H_Y = torch.nn.Parameter(H_Y_init)

    # ==================== Forward / Loss ====================
    def _reconstruct(self, which='both'):
        if which == 'X':
            return self._build_W('X') @ self.H_X  # (g, n_X)
        elif which == 'Y':
            return self._build_W('Y') @ self.H_Y  # (g, n_Y)
        else:
            return (self._build_W('X') @ self.H_X,
                    self._build_W('Y') @ self.H_Y)

    def _loss(self, X, Y):
        recon_X, recon_Y = self._reconstruct('both')

        loss_X = torch.sum((X.T - recon_X) ** 2)
        loss_Y = torch.sum((Y.T - recon_Y) ** 2)

        # 共享字典正则（始终存在）
        reg_shared = self.lam_shared * torch.sum(torch.abs(self.W_shared))
        # 私有字典正则；n_private=0 时该项恒为 0
        reg_private = self.lam_private * self._private_norm()

        total = loss_X + loss_Y + reg_shared + reg_private
        return total, {'recon_X': loss_X, 'recon_Y': recon_Y}

    # ==================== Fit ====================
    def fit(self, X, Y):
        """
        X : (n_X, n_genes) numpy array
        Y : (n_Y, n_genes) numpy array
        """
        if isinstance(X, np.ndarray):
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        else:
            X_t, Y_t = X, Y

        self._init_params(X_t, Y_t)

        optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=30, min_lr=1e-6)

        self.loss_history = []
        prev_loss = float('inf')

        iterator = range(self.max_iter)
        if self.verbose:
            if not _HAS_TQDM:
                warnings.warn(
                    "verbose=True but tqdm is not installed. "
                    "Install with `pip install tqdm` for a progress bar.",
                    RuntimeWarning)
                iterator = iterator
            else:
                iterator = tqdm(
                    iterator, desc='BiDictionaryNMF',
                    unit='iter', leave=True,
                    dynamic_ncols=True)

        for it in iterator:
            optimizer.zero_grad()

            total_loss, details = self._loss(X_t, Y_t)
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.W_shared.clamp_(min=0)
                if self._has_private():
                    self.W_X.clamp_(min=0)
                    self.W_Y.clamp_(min=0)
                self.H_X.clamp_(min=0)
                self.H_Y.clamp_(min=0)

            loss_val = total_loss.item()
            self.loss_history.append(loss_val)
            scheduler.step(loss_val)

            if self.verbose and _HAS_TQDM:
                iterator.set_postfix(
                    loss=f"{loss_val:.4e}",
                    #X=f"{details['recon_X'].item():.2e}",
                    #Y=f"{details['recon_Y'].item():.2e}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}")

            rel_improve = (prev_loss - loss_val) / max(abs(prev_loss), 1e-10)
            if 0 < rel_improve < self.tol:
                if self.verbose:
                    iterator.write(f"Early stop at iter {it} "
                                   f"(rel_improve={rel_improve:.2e})")
                break
            prev_loss = loss_val

        self._fitted = True
        if self.verbose and _HAS_TQDM:
            private_msg = (f"||W_X||_1={self.W_X.abs().sum():.1f} "
                           f"||W_Y||_1={self.W_Y.abs().sum():.1f}")
            iterator.write(
                f"Done. W_shared={tuple(self.W_shared.shape)} | "
                f"{private_msg}")
        elif self.verbose:
            print("Done.")
            print(f"  W_shared: {self.W_shared.shape} | "
                  f"||W_X||_1={self.W_X.abs().sum():.1f} "
                  f"||W_Y||_1={self.W_Y.abs().sum():.1f}")

        return self

    # ==================== Transform ====================
    def transform(self, Z, batch='X'):
        """
        对新数据 Z 计算 compositions（不更新 W）

        Parameters
        ----------
        Z : (n_cells, n_genes) numpy array
        batch : 'X' or 'Y'

        Returns
        -------
        H : (n_total, n_cells) numpy array
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet.")

        if isinstance(Z, np.ndarray):
            Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)
        else:
            Z_t = Z

        with torch.no_grad():
            W = self._build_W(batch)

            H = torch.relu(torch.linalg.pinv(W) @ Z_t.T).clone().detach()
            H.requires_grad_(True)
            opt = Adam([H], lr=1e-2)

            for _ in range(200):
                opt.zero_grad()
                recon = W @ H
                loss = torch.sum((Z_t.T - recon) ** 2)
                loss.backward()
                opt.step()
                H.clamp_(min=0)

            return H.cpu().numpy()

    # ==================== 获取共享部分 ====================
    def get_shared_compositions(self):
        """
        Returns
        -------
        H_X_shared : (n_shared, n_X) numpy
        H_Y_shared : (n_shared, n_Y) numpy
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet.")

        with torch.no_grad():
            H_X_s = self.H_X[:self.n_shared, :].cpu().numpy()
            H_Y_s = self.H_Y[:self.n_shared, :].cpu().numpy()
        return H_X_s, H_Y_s

    def get_shared_genes(self, gene_names=None, top_k=10):
        """
        返回每个共享 GP 的 top 基因索引或名称
        """
        if not self._fitted:
            raise ValueError("Model not fitted yet.")

        with torch.no_grad():
            W_s = self.W_shared.cpu().numpy()

        results = []
        for k in range(self.n_shared):
            idx = np.argsort(W_s[:, k])[::-1][:top_k]
            if gene_names is not None:
                results.append([gene_names[i] for i in idx])
            else:
                results.append(idx)
        return results

    # ==================== 模型保存 / 加载 ====================
    def save(self, path, save_optimizer=False):
        """
        保存模型到文件。

        保存内容：模型配置、学习到的参数（W_shared/W_X/W_Y/H_X/H_Y）、
        loss 历史、拟合状态。默认不保存优化器状态（纯推理/部署用）；
        如需断点续训可设置 ``save_optimizer=True``（需在 fit 中额外保留
        optimizer 引用，当前实现不默认保存）。

        Parameters
        ----------
        path : str
            输出文件路径，建议以 ``.pt`` 结尾。
        save_optimizer : bool
            是否保存优化器状态（默认 False）。
        """
        config = {k: getattr(self, k) for k in self._CONFIG_KEYS}

        state = {
            'config': config,
            'model_state_dict': self.state_dict(),
            'loss_history': self.loss_history,
            'fitted': self._fitted,
            'torch_version': torch.__version__,
        }
        if save_optimizer:
            warnings.warn(
                "save_optimizer=True but no optimizer reference is kept. "
                "Ignoring.", RuntimeWarning)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path, map_location=None, strict=True):
        """
        从文件加载模型。

        Parameters
        ----------
        path : str
            模型文件路径。
        map_location : str or torch.device, optional
            指定加载到的设备，例如 ``'cpu'``。为 None 时按保存时的 device
            字段选择设备。
        strict : bool
            是否严格匹配 state_dict 的 keys（默认 True）。

        Returns
        -------
        BiDictionaryNMF
            已加载并切换到对应设备的模型实例。
        """
        checkpoint = torch.load(path, map_location=map_location)

        config = checkpoint['config']
        device = config.get('device', None)
        if map_location is not None:
            device = map_location

        model = cls(**config)
        if device is not None:
            model.device = device

        # 重新构建参数容器：从 state_dict 中读出真实形状，用零张量占位，
        # 随后由 load_state_dict 填入实际数值。
        state_dict = checkpoint['model_state_dict']
        g, _n_shared = state_dict['W_shared'].shape
        n_X_samples = state_dict['H_X'].shape[1]
        n_Y_samples = state_dict['H_Y'].shape[1]

        if _n_shared != model.n_shared:
            raise ValueError(
                f"Saved W_shared has {_n_shared} columns but config "
                f"n_shared={model.n_shared}.")

        # 行数（基因数）与 H 的第一维（= n_total）必须自洽。
        # 注意：n_private=0 时 W_X / W_Y 形状为 (g, 0)，需单独按维度数判断。
        def _cols(t):
            return t.shape[1] if t.ndim >= 2 else t.shape[0]

        assert _cols(state_dict['W_X']) == model.n_private
        assert _cols(state_dict['W_Y']) == model.n_private
        assert state_dict['H_X'].shape[0] == model.n_total
        assert state_dict['H_Y'].shape[0] == model.n_total

        model.W_shared = torch.nn.Parameter(
            torch.zeros(g, model.n_shared, device=model.device))
        model.W_X = torch.nn.Parameter(
            torch.zeros(g, model.n_private, device=model.device))
        model.W_Y = torch.nn.Parameter(
            torch.zeros(g, model.n_private, device=model.device))
        model.H_X = torch.nn.Parameter(
            torch.zeros(model.n_total, n_X_samples, device=model.device))
        model.H_Y = torch.nn.Parameter(
            torch.zeros(model.n_total, n_Y_samples, device=model.device))

        model.load_state_dict(state_dict, strict=strict)

        model.loss_history = checkpoint.get('loss_history', [])
        model._fitted = checkpoint.get('fitted', True)

        if device is not None:
            model.to(device)

        return model

    # ==================== 属性访问 ====================
    @property
    def W_shared_np(self):
        return self.W_shared.detach().cpu().numpy()

    @property
    def W_X_np(self):
        return self.W_X.detach().cpu().numpy()

    @property
    def W_Y_np(self):
        return self.W_Y.detach().cpu().numpy()

    @property
    def H_X_np(self):
        return self.H_X.detach().cpu().numpy()

    @property
    def H_Y_np(self):
        return self.H_Y.detach().cpu().numpy()
