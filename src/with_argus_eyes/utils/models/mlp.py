"""
mlp.py
-------

This module defines a configurable multilayer perceptron (MLP) regressor using
PyTorch.  It is designed to support experiments with different network
architectures, loss functions, dropout rates, learning rates, and regularisation
strengths.  It also records training and validation losses at each epoch to
facilitate learning curve visualisation.

Dependencies
------------
This module requires PyTorch (``torch``) to be installed.  If you do not have
PyTorch available, install it by following the instructions at
https://pytorch.org/get-started/locally/

Example
-------

.. code-block:: python

    from mlp import MLPRegressor
    import numpy as np

    # Generate dummy data
    X_train = np.random.rand(100, 20)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 20)
    y_val = np.random.rand(20)

    mlp = MLPRegressor(
        input_dim=20,
        hidden_dims=(64, 32),
        dropout=0.1,
        loss="huber",
        huber_delta=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=50,
        batch_size=32,
        verbose=True,
    )

    mlp.fit(X_train, y_train, X_val, y_val)
    preds = mlp.predict(X_val)
    print("RMSE:", ((preds - y_val)**2).mean()**0.5)
    print("Training history:", mlp.history)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)
COSMETIC_SCHEDULER_ALIASES: Dict[str, str] = {
    "plateau": "ReduceLROnPlateau",
    "reduceonplateau": "ReduceLROnPlateau",
    "reducelronplateau": "ReduceLROnPlateau",
    "step": "StepLR",
    "steplr": "StepLR",
    "multi_step": "MultiStepLR",
    "multistep": "MultiStepLR",
    "cosine": "CosineAnnealingLR",
    "cosineannealing": "CosineAnnealingLR",
    "cosinewarm": "CosineAnnealingWarmRestarts",
    "cosineannealingwarmrestarts": "CosineAnnealingWarmRestarts",
    "onecycle": "OneCycleLR",
    "onecyclelr": "OneCycleLR",
    "exponential": "ExponentialLR",
    "exp": "ExponentialLR",
    "linear": "LinearLR",
    "polynomial": "PolynomialLR",
    "poly": "PolynomialLR",
}


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "PyTorch is required to use MLPRegressor. Please install torch following the instructions at https://pytorch.org/."
    ) from exc


class _RegressionMLP(nn.Module):
    """Internal PyTorch module implementing the MLP architecture."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(prev_dim, hid))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hid
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        return self.net(x).squeeze(-1)  # shape: (batch_size,)


# ---------------------------------------------------------------------------
# Top-level KL loss (picklable)
# ---------------------------------------------------------------------------

def kl_bernoulli_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence between Bernoulli distributions parameterised by targets and logits.

    We interpret the scalar target ``y`` as the probability of class 1 in a
    two-class distribution (y, 1−y). The raw model outputs (logits) are passed
    through a sigmoid to obtain predicted probabilities. We then build
    [p, 1-p] and [t, 1-t] and compute KL(target || pred) with batch mean
    reduction.
    """
    eps = 1e-7
    # Convert raw logits to probabilities
    p = torch.sigmoid(preds).clamp(eps, 1 - eps)
    t = targets.clamp(eps, 1 - eps)
    # Two-class distributions [p, 1−p] and [t, 1−t]
    q = torch.stack([p, 1.0 - p], dim=1)
    target_dist = torch.stack([t, 1.0 - t], dim=1)
    # log probabilities for KLDivLoss expect log(P)
    log_q = torch.log(q)
    # KLDivLoss computes KL(target || input). Use 'batchmean' reduction.
    return torch.nn.functional.kl_div(log_q, target_dist, reduction="batchmean")


@dataclass
class MLPRegressor:
    """A configurable multilayer perceptron regressor built with PyTorch.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    hidden_dims : tuple of int, optional
        Sizes of the hidden layers.  For example ``(512, 256, 64)`` constructs
        three hidden layers with those widths.  An empty tuple defaults to a
        single linear layer (i.e., no hidden layers).
    dropout : float, optional
        Dropout probability applied after each hidden layer.  Set to 0.0 to
        disable dropout.
    loss : {'mse', 'mae', 'huber', 'kl'}, optional
        Which loss function to optimise during training.
    huber_delta : float, optional
        Delta parameter for the Huber loss.  Only used when ``loss='huber'``.
    lr : float, optional
        Learning rate for the optimiser.
    weight_decay : float, optional
        Weight decay (ℓ₂ regularisation) applied by the optimiser.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Number of examples per mini-batch.
    verbose : bool, optional
        Whether to log per-epoch training progress.
    device : str or torch.device, optional
        Device on which to run the model.  If None, defaults to the current
        PyTorch default device.
    seed : int, optional
        Random seed for reproducible weight initialisation.
    scheduler : dict | str | sequence, optional
        Learning-rate scheduler specification.  You can pass:
        - a dictionary with ``name``/``type`` plus kwargs
        - a string with the scheduler class name (no kwargs)
        - a tuple/list ``(name, kwargs_dict)``
        The class must exist inside ``torch.optim.lr_scheduler``.  Scheduler
        ``step`` is called once per epoch after validation.  Case-insensitive
        aliases such as ``"plateau"`` (``ReduceLROnPlateau``) and ``"step"``
        (``StepLR``) are supported.  Use ``"none"`` or ``null`` to disable.
    """

    input_dim: int
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.0
    loss: str = "mse"
    huber_delta: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 100
    batch_size: int = 32
    verbose: bool = False
    device: str | torch.device | None = None
    seed: int = 42
    scheduler: Dict[str, Any] | None = None
    history: dict[str, List[float]] = field(
        default_factory=lambda: {"train_loss": [], "val_loss": [], "lr": []},
        init=False,
    )
    _scheduler_name: str | None = field(default=None, init=False)
    _scheduler_requires_metric: bool = field(default=False, init=False)
    _scheduler: Any | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Determine device (and ensure it's valid)
        self.device = self._resolve_device(self.device)

        # Build the network
        self.model = _RegressionMLP(self.input_dim, self.hidden_dims, self.dropout).to(self.device)

        # Configure loss function
        loss_lower = self.loss.lower()
        if loss_lower == "mse":
            self._criterion = nn.MSELoss()
        elif loss_lower == "mae":
            self._criterion = nn.L1Loss()
        elif loss_lower == "huber":
            # SmoothL1Loss implements the Huber loss in PyTorch, parameter beta controls the
            # transition point between L1 and L2 losses.  In PyTorch <=1.9 the parameter is
            # 'beta' in later versions it's 'delta'.  To be safe we pass both if available.
            try:
                self._criterion = nn.SmoothL1Loss(beta=self.huber_delta)
            except TypeError:
                self._criterion = nn.SmoothL1Loss(delta=self.huber_delta)
        elif loss_lower in {"kl", "kld", "kl_div"}:
            # Use top-level KL loss to keep the model picklable
            self._criterion = kl_bernoulli_loss
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

        # Create optimiser
        self._optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._scheduler = self._build_scheduler()

    def _build_scheduler(self) -> Any | None:
        """Instantiate an LR scheduler if the user provided configuration."""
        if not self.scheduler:
            return None

        scheduler_conf = self._normalise_scheduler_config(self.scheduler)
        if scheduler_conf is None:
            return None
        scheduler_name, scheduler_cfg = scheduler_conf

        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name, None)
        if scheduler_cls is None:
            raise ValueError(
                f"Unknown scheduler '{scheduler_name}'. Expected a class from torch.optim.lr_scheduler."
            )

        self._scheduler_name = scheduler_name
        self._scheduler_requires_metric = scheduler_name.lower() == "reducelronplateau"

        try:
            scheduler = scheduler_cls(self._optim, **scheduler_cfg)
        except TypeError as exc:
            raise TypeError(
                f"Failed to initialise scheduler '{scheduler_name}' with arguments {scheduler_cfg}"
            ) from exc

        return scheduler

    @staticmethod
    def _normalise_scheduler_config(raw_cfg: Any) -> Tuple[str, Dict[str, Any]] | None:
        """Support multiple scheduler config formats."""
        def resolve_name(raw_name: Any) -> str | None:
            if raw_name is None:
                return None
            name = str(raw_name).strip()
            if name.lower() in {"", "none", "null", "off"}:
                return None
            return COSMETIC_SCHEDULER_ALIASES.get(name.lower(), name)

        if isinstance(raw_cfg, dict):
            scheduler_cfg = dict(raw_cfg)
            scheduler_name = scheduler_cfg.pop("name", None) or scheduler_cfg.pop("type", None)
            scheduler_name = resolve_name(scheduler_name)
            if scheduler_name is None:
                return None
            if scheduler_cfg and not isinstance(scheduler_cfg, dict):
                raise TypeError("Scheduler kwargs must form a dict.")
        elif isinstance(raw_cfg, (list, tuple)):
            if len(raw_cfg) == 0:
                raise ValueError("Scheduler sequence cannot be empty.")
            scheduler_name = resolve_name(raw_cfg[0])
            if scheduler_name is None:
                return None
            scheduler_cfg = raw_cfg[1] if len(raw_cfg) > 1 else {}
            if scheduler_cfg is None:
                scheduler_cfg = {}
            if not isinstance(scheduler_cfg, dict):
                raise TypeError("Scheduler sequence second element must be a dict of kwargs.")
        elif isinstance(raw_cfg, str):
            scheduler_name = resolve_name(raw_cfg)
            if scheduler_name is None:
                return None
            scheduler_cfg = {}
        else:
            raise TypeError("Scheduler config must be dict, string, or (name, kwargs) sequence.")

        if not isinstance(scheduler_name, str):
            raise TypeError("Scheduler name must be a string.")

        return scheduler_name, scheduler_cfg

    def _default_device(self) -> torch.device:
        """Pick the best-available default device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        """Normalise string/None devices and gracefully fall back if unavailable."""
        if isinstance(device, torch.device):
            resolved = device
        elif isinstance(device, str):
            try:
                resolved = torch.device(device)
            except (TypeError, RuntimeError):
                logger.warning("Invalid device specification '%s'; falling back to default.", device)
                resolved = self._default_device()
        else:
            resolved = self._default_device()

        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                resolved = torch.device("cpu")
            elif resolved.index is not None:
                device_count = torch.cuda.device_count()
                if resolved.index >= device_count:
                    logger.warning(
                        "CUDA device %s unavailable (only %d devices). Falling back to cuda:0.",
                        resolved,
                        device_count,
                    )
                    if device_count > 0:
                        resolved = torch.device("cuda:0")
                    else:
                        resolved = torch.device("cpu")

        return resolved

    def _ensure_model_device(self, device: torch.device | str | None = None) -> torch.device:
        """
        Make sure ``self.model`` and future tensors live on the same, valid device.

        This is especially important after loading pickled/joblib artefacts where
        the stored device might not exist on the current host.
        """
        target = self._resolve_device(device if device is not None else self.device)
        model_module = getattr(self, "model", None)
        if model_module is not None:
            try:
                model_module.to(target)
            except RuntimeError as exc:
                logger.warning("Failed to move model to %s (%s). Falling back to CPU.", target, exc)
                target = torch.device("cpu")
                model_module.to(target)
        self.device = target
        return target

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "MLPRegressor":
        """Train the MLP regressor on the provided dataset.

        If validation data is supplied, validation loss will be recorded at
        each epoch.  Both training and validation losses are stored in the
        ``history`` attribute.
        """
        device = self._ensure_model_device()

        # Convert inputs to torch tensors
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        if X_val is not None and y_val is not None:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)
        else:
            X_val_t = y_val_t = None

        num_train = X_train_t.shape[0]
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Shuffle training data indices
            perm = torch.randperm(num_train, device=self.device)
            train_loss_epoch = 0.0
            self.model.train()
            for start in range(0, num_train, self.batch_size):
                end = min(start + self.batch_size, num_train)
                idx = perm[start:end]
                xb = X_train_t[idx]
                yb = y_train_t[idx]
                # Forward pass
                preds = self.model(xb)
                loss = self._criterion(preds, yb)
                # Backpropagation
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()
                train_loss_epoch += loss.item() * (end - start)
            # Average training loss over all samples
            train_loss_epoch /= num_train
            self.history["train_loss"].append(train_loss_epoch)

            # Validation loss
            if X_val_t is not None and y_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_t)
                    val_loss = self._criterion(val_preds, y_val_t).item()
                self.history["val_loss"].append(val_loss)
            else:
                val_loss = None

            # Step LR scheduler if configured
            if self._scheduler is not None:
                metric_for_scheduler = val_loss if val_loss is not None else train_loss_epoch
                if self._scheduler_requires_metric and val_loss is None:
                    logger.warning(
                        "Scheduler '%s' requires a validation metric, but none was provided. Falling back to training loss.",
                        self._scheduler_name,
                    )
                if self._scheduler_requires_metric:
                    self._scheduler.step(metric_for_scheduler)
                else:
                    self._scheduler.step()
                current_lr = self._optim.param_groups[0]["lr"]
                self.history.setdefault("lr", []).append(current_lr)
            else:
                # Keep lr history in sync with epochs even without a scheduler
                self.history.setdefault("lr", []).append(self._optim.param_groups[0]["lr"])

            if self.verbose:
                if val_loss is not None:
                    logger.info(
                        "Epoch %3d/%d | Train loss: %.6f | Val loss: %.6f",
                        epoch,
                        self.epochs,
                        train_loss_epoch,
                        val_loss,
                    )
                else:
                    logger.info(
                        "Epoch %3d/%d | Train loss: %.6f", epoch, self.epochs, train_loss_epoch
                    )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous outputs for the given input features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, input_dim)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        device = self._ensure_model_device()
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw_preds = self.model(X_t)
            # For KL divergence loss we interpret the outputs as logits and
            # convert them to probabilities via sigmoid before returning.  For
            # all other losses, return the raw regression outputs.
            if self.loss.lower() in {"kl", "kld", "kl_div"}:
                preds = torch.sigmoid(raw_preds)
            else:
                preds = raw_preds
            preds_np = preds.cpu().numpy()
        return preds_np.astype(float)