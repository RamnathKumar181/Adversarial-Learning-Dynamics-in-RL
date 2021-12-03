import torch
import torch.nn.functional as F


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()


def JSD(net_1_logits, net_2_logits):
    net_1_probs = F.softmax(net_1_logits, dim=1)
    net_2_probs = F.softmax(net_2_logits, dim=1)

    m = 0.5 * (net_1_probs + net_2_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="batchmean")
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="batchmean")

    return (0.5 * loss)
