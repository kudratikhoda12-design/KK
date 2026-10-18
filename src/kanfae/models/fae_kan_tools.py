"""D104: pruning and symbolic extraction for the KAN blocks of a fitted fae_kan NeuralAFT (both two_checkpoint states).

States: "model" (objective-minimum state: latent and reconstruction) and "risk" (the two_checkpoint risk state, which
alone produces eta and sigma). Block inputs are the TRAINING inputs of each block, computed in eval mode (BatchNorm
running statistics fixed).
  uncentred  KANBlock.prune: edge score mean|phi| < theta, then hidden-node pruning (KAN-AFT / CoxKAN reference)
  centred    D54 score mean|phi - mean(phi)| < theta; the pruned edge's mean is absorbed into its output bias
"""
import numpy as np
import torch

from ..symbolic.fit import fit_symbolic
from .kan_aft import _torch_formula


def states(m):
    out = {"model": m.model}
    if getattr(m, "risk_model", None) is not None:
        out["risk"] = m.risk_model
    return out


@torch.no_grad()
def block_inputs(net, c):
    """Raw (pre-normalisation) inputs of the encoder and decoder KAN blocks of one state."""
    net.eval()
    h = net.flin(c)
    return {"encoder": h, "decoder": net.encoder(h)}


def train_tensor(m, obs_tr):
    return m._tensors(m.prepare(obs_tr))["inp"]


@torch.no_grad()
def edge_scores(block, x):
    """Per layer: (uncentred, centred) edge scores on the block's training inputs; masked edges score 0."""
    block.eval()
    out = []
    for layer, xi in zip(block.layers, block.layer_inputs(x)):
        phi = layer.edge_activations(xi)
        out.append((phi.abs().mean(0), (phi - phi.mean(0)).abs().mean(0)))
    return out


def active_edges(block):
    return int(sum(int(l.mask.sum()) for l in block.layers))


@torch.no_grad()
def prune_uncentred(block, x, theta=1e-2):
    block.eval()
    return block.prune(x, theta)


@torch.no_grad()
def prune_centred(block, x, theta=1e-2):
    block.eval()
    n = 0
    for layer, xi in zip(block.layers, block.layer_inputs(x)):
        phi = layer.edge_activations(xi)
        mean = phi.mean(0)
        sel = ((phi - mean).abs().mean(0) < theta) & (layer.mask > 0)
        layer.bias += (mean * sel).sum(0)
        layer.mask[sel] = 0.0
        n += int(sel.sum())
    return {"edges_pruned": n, "nodes_pruned": 0}


def prune_member(m, obs_tr, method, theta=1e-2):
    """Prune both KAN blocks of both states of member m in place; returns counts per state and block."""
    c = train_tensor(m, obs_tr)
    f = prune_uncentred if method == "uncentred" else prune_centred
    info = {}
    for sname, net in states(m).items():
        x = block_inputs(net, c)
        for bname in ("encoder", "decoder"):
            info[f"{sname}_{bname}"] = f(getattr(net, bname), x[bname], theta)
            info[f"{sname}_{bname}_active_after"] = active_edges(getattr(net, bname))
    return info


def symbolic_block(block, x, use_pysr=False, pysr_budget=None, edges=None, substitute=True):
    """Fit every active edge (or the given (layer, i, o) list) of a KAN block on its training inputs; substitute
    accepted formulas (others keep their spline). Returns one record per edge."""
    block.eval()
    with torch.no_grad():
        ins = block.layer_inputs(x)
    recs = []
    for li, (layer, xin) in enumerate(zip(block.layers, ins)):
        for i in range(layer.n_in):
            for o in range(layer.n_out):
                if layer.mask[i, o] == 0 or (edges is not None and (li, i, o) not in edges):
                    continue
                xs = xin[:, i].detach().numpy()

                def func(z, layer=layer, i=i, o=o):
                    Z = torch.zeros(len(z), layer.n_in)
                    Z[:, i] = torch.tensor(np.asarray(z, float))
                    with torch.no_grad():
                        return layer.edge_activations(Z)[:, i, o].numpy()
                ys = func(xs)
                res = fit_symbolic(xs, ys, func, use_pysr=use_pysr, pysr_budget=pysr_budget)
                d = res.to_dict()
                recs.append(dict(layer=li, i=i, o=o, l1=float(np.mean(np.abs(ys))), **{k: d[k] for k in
                            ("accepted", "stage", "formula", "r2_val", "complexity") if k in d}))
                if substitute and res.accepted:
                    layer.symbolic[(i, o)] = _torch_formula(res)
    return recs
