import torch
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Iterator

class Muon(Optimizer):
    """
    Implements the Muon optimization algorithm for linear layers.

    Muon uses a geometric approach to optimization, specifically addressing
    how changes in weight matrices affect neural network behavior.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        ns_iters (int, optional): number of Newton-Schulz iterations (default: 5)
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    def __init__(self,
                 params: Iterator[torch.nn.Parameter],
                 lr: float = 1e-3,
                 ns_iters: int = 5,
                 momentum: float = 0.9,
                 weight_decay: float = 0):

        defaults = dict(lr=lr, ns_iters=ns_iters, momentum=momentum, weight_decay=weight_decay)
        super(Muon, self).__init__(params, defaults)

    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations to approximate orthogonalization.

        This function applies the polynomial f(X) = (3X - X^3)/2 repeatedly to a normalized matrix,
        which gradually forces all singular values to 1 while preserving singular vectors.

        Args:
            X (torch.Tensor): Input matrix to orthogonalize
            num_iters (int): Number of Newton-Schulz iterations

        Returns:
            torch.Tensor: Orthogonalized matrix
        """
        # First, normalize the input matrix to get spectral norm close to 1
        # We use Frobenius norm as a simple approximation for initialization
        norm = torch.norm(X, p='fro')
        if norm < 1e-8:
            return X  # Avoid division by zero

        X = X / norm

        # Apply Newton-Schulz iterations
        for _ in range(num_iters):
            X = (3 * X - torch.matmul(torch.matmul(X, X), X)) / 2

        return X

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss

        Returns:
            Optional[float]: Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            ns_iters = group['ns_iters']
            momentum_factor = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Handle weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # Initialize momentum buffer if needed
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                # Get momentum buffer
                momentum_buffer = state['momentum_buffer']

                # Update momentum buffer with current gradient
                momentum_buffer.mul_(momentum_factor).add_(grad, alpha=1 - momentum_factor)

                # Only apply Muon updates to matrices (linear layers)
                if len(p.shape) == 2:
                    # Get input and output dimensions for normalization
                    d_in, d_out = p.shape

                    # Use the momentum buffer for orthogonalization
                    ortho_grad = self.newton_schulz_orthogonalize(momentum_buffer, ns_iters)

                    # Scale by sqrt(d_in * d_out) / |G|_F as per Muon's formula
                    grad_norm = torch.norm(momentum_buffer, p='fro')
                    if grad_norm > 1e-8:  # Avoid division by zero
                        scaling = (d_in * d_out)**0.5 / grad_norm
                        update = ortho_grad * scaling

                        # Apply the update
                        p.add_(update, alpha=-lr)

                else:
                    # For non-matrix parameters (biases, etc.), use standard update with momentum
                    p.add_(momentum_buffer, alpha=-lr)

        return loss