from __future__ import annotations
import torch


def _coalesced_sparse(indices: torch.Tensor, values: torch.Tensor, num_nodes: int, dtype) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), device=indices.device, dtype=dtype
    ).coalesce()


def _with_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Add one self loop per node, then coalesce duplicate edges without PyG."""
    diag = torch.arange(num_nodes, device=edge_index.device, dtype=edge_index.dtype)
    loops = torch.stack([diag, diag], dim=0)
    ei = torch.cat([edge_index, loops], dim=1)
    # Coalescing a unit-valued sparse matrix gives unique edge coordinates.
    ones = torch.ones(ei.size(1), device=ei.device, dtype=torch.float32)
    sp = torch.sparse_coo_tensor(ei, ones, (num_nodes, num_nodes)).coalesce()
    return sp.indices()


def lazy_random_walk(edge_index: torch.Tensor, num_nodes: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """W_lazy = 1/2 (I + D_tilde^{-1} A_tilde), A_tilde=A+I.

    The returned sparse COO matrix acts as ``H_out = W_lazy @ H``.
    ``edge_index`` follows PyG's source->target convention, while sparse
    matrix rows are targets and columns are sources.  Construction remains
    sparse and therefore scales to ogbn-arxiv.
    """
    ei = edge_index.to(device) if device is not None else edge_index
    ei = _with_self_loops(ei, num_nodes)
    src, dst = ei[0], ei[1]
    deg = torch.bincount(dst, minlength=num_nodes).to(device=ei.device, dtype=dtype).clamp_min_(1.0)
    rw_vals = 0.5 / deg[dst]

    # + 0.5 I for laziness.  Duplicate diagonal coordinates are intentionally
    # summed by sparse coalescing.
    diag = torch.arange(num_nodes, device=ei.device, dtype=ei.dtype)
    rows = torch.cat([dst, diag])
    cols = torch.cat([src, diag])
    vals = torch.cat([rw_vals, torch.full((num_nodes,), 0.5, device=ei.device, dtype=dtype)])
    return _coalesced_sparse(torch.stack([rows, cols], dim=0), vals, num_nodes, dtype)


def rw_with_self_loops(edge_index: torch.Tensor, num_nodes: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """D_tilde^{-1} A_tilde as a sparse COO matrix."""
    ei = edge_index.to(device) if device is not None else edge_index
    ei = _with_self_loops(ei, num_nodes)
    src, dst = ei[0], ei[1]
    deg = torch.bincount(dst, minlength=num_nodes).to(device=ei.device, dtype=dtype).clamp_min_(1.0)
    vals = 1.0 / deg[dst]
    return _coalesced_sparse(torch.stack([dst, src], dim=0), vals, num_nodes, dtype)


def spmm(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(mat, x)
