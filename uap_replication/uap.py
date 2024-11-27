# PyTorch DeepFool Implementation
# Adapted from github.com/LTS4/universal/ and github.com/qilong-zhang/Pytorch_Universal-adversarial-perturbation.
# For method documentation, see Python implementation in the repos above.

import numpy as np
import torch
from torch.utils.data import DataLoader


def estimate_fooling_rate(dataset, clf, v):
    fooled = 0
    with torch.no_grad():
        for img in DataLoader(dataset):
            if clf(img).argmax() != clf(img+v).argmax():
                fooled += 1
    return fooled / len(dataset)


# See Algorithm 2 in arxiv.org/pdf/1511.04599.
def deepfool(img, clf, num_classes=10, overshoot=0.02, max_iter=10):
    # Get indices of the n likeliest classes for img (where n = num_classes).
    I = clf(img).argsort()[::-1][0:num_classes]

    # Initialize values. r will be the final output.
    pert_img = torch.Tensor(img, requires_grad=True)
    r = torch.Tensor(np.zeros(img.shape))
    i = 0

    while i < max_iter:
        # Compute label of pert_img.
        # If this is different from the label of img, we're done.
        out = clf(pert_img)
        k_i = out.argmax()
        if k_i != I[0]:
            break

        # Let grad_0 be the gradient of pert_img's I[0]th activiation. 
        clf.zero_grad()
        out[0, I[0]].backward(retain_graph=True)
        grad_0 = pert_img.grad.detach().clone()

        # Compute r_i as in the paper.
        w_badness = np.inf
        w = None

        for k in range(1, num_classes):
            clf.zero_grad()
            out[0, I[k]].backward(retain_graph=True)
            grad_k = pert_img.grad.detach().clone()

            w_k = grad_k - grad_0
            f_k = out[0, I[k]] - out[0, I[0]]
            w_k_badness = abs(f_k) / np.linalg.norm(w_k)

            if w_k_badness < w_badness:
                w_badness = w_k_badness
                w = w_k

        r_i = (w_badness + 1e-4) * w / np.linalg.norm(w)
        r = r + r_i
        pert_img = torch.Tensor(img + (1+overshoot)*r, requires_grad=True)

        i += 1
    
    return (1 + overshoot) * r, i


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 of radius xi
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v


# See Algorithm 1 in arxiv.org/pdf/1610.08401.
def compute_uap(dataset, clf, delta=0.2, max_iter=np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    est_fooling_rate = 0
    v = 0
    i = 0

    while est_fooling_rate < 1-delta and  i < max_iter:
        print(f"Deepfool Iteration {i}")
        i += 1

        loader = DataLoader(dataset, shuffle=True)
        for img in loader:
            orig_cls = int(clf(img).argmax())
            pert_cls = int(clf(img+v).argmax())

            if orig_cls == pert_cls:
                delta_v, iterations = deepfool(img+v, clf, num_classes, overshoot, max_iter_df)
                if iterations < max_iter_df-1:
                    v += delta_v
                    v = proj_lp(v, xi, p)

        rate = estimate_fooling_rate(dataset, clf, v)
        print(f"Estimated Fooling Rate: {rate}")

    return v
