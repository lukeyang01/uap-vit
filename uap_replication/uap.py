# PyTorch DeepFool Implementation
# Adapted from github.com/LTS4/universal/ and github.com/qilong-zhang/Pytorch_Universal-adversarial-perturbation.
# For method documentation, see Python implementation in the repos above.

import numpy as np
import torch
from torch.utils.data import DataLoader


def estimate_fooling_rate(dataset, clf, v):
    fooled = 0
    with torch.no_grad():
        for _, (img, _) in enumerate(DataLoader(dataset)):
            if clf(img).argmax() != clf(img+v).argmax():
                fooled += 1
    return fooled / len(dataset)

def compute_validation_rate(dataset, clf, v):
    correct_pert = 0
    with torch.no_grad():
        for _, (img, truth) in enumerate(DataLoader(dataset)):
            truth = int(truth)
            if clf(img+v).argmax() == truth:
                correct_pert += 1
    return correct_pert / len(dataset)


# See Algorithm 2 in arxiv.org/pdf/1511.04599.
def deepfool(img, clf, num_classes=5, overshoot=0.02, max_iter=50):
    # Get indices of the n likeliest classes for img (where n = num_classes).
    I = clf(img)[0].argsort().flip(0)[0:num_classes]

    # Initialize values. r will be the final output.
    r = np.zeros(img.shape).astype(np.float64)
    i = 0

    while i < max_iter:
        pert_img = (img + (1+overshoot)*torch.Tensor(r)).requires_grad_()

        # Compute label of pert_img.
        # If this is different from the label of img, we're done.
        out = clf(pert_img)
        if out.argmax() != I[0]:
            break

        # j[k] is the gradient of pert_img's I[k]th activiation.
        j = torch.autograd.functional.jacobian(lambda t: clf(t)[0][I], pert_img)
        j = j.detach().numpy().copy()

        # Compute r_i as in the paper.
        w_badness = np.inf
        w = None
        for k in range(1, num_classes):
            w_k = j[k] - j[0]
            f_k = (out[0, I[k]] - out[0, I[0]]).detach().numpy()
            w_k_badness = abs(f_k) / np.linalg.norm(w_k)
            if w_k_badness < w_badness:
                w_badness = w_k_badness
                w = w_k

        r_i = (w_badness + 1e-4) * w / np.linalg.norm(w)
        r += r_i
        i += 1

    return torch.Tensor((1 + overshoot) * r), i


def proj_lp(v, xi, p):
    # Project on the lp ball centered at 0 of radius xi
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten()))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v


# See Algorithm 1 in arxiv.org/pdf/1610.08401.
def compute_uap(dataset, clf, cb=None, delta=0.2, max_iter=np.inf, xi=10, p=np.inf, num_classes=5, overshoot=0.02, max_iter_df=50):
    est_fooling_rate = 0
    v = 0
    i = 0

    while est_fooling_rate < 1-delta and  i < max_iter:
        print(f"UAP ITERATION {i}")
        i += 1

        loader = DataLoader(dataset, shuffle=True)
        convergences = 0
        deepfools = 0
        for count, (img, _) in enumerate(loader, start=1):
            orig_cls = clf(img).argmax()
            pert_cls = clf(img+v).argmax()

            if orig_cls == pert_cls:
                deepfools += 1
                delta_v, iterations = deepfool(img+v, clf, num_classes, overshoot, max_iter_df)
                if iterations < max_iter_df-1:
                    convergences += 1
                    v += delta_v
                    v = proj_lp(v, xi, p)

            if count % 10 == 0:
                print(f"{convergences} Convergences / {deepfools} Deepfools / {count} Images")

        if cb:
            cb(i, v)

        rate = estimate_fooling_rate(dataset, clf, v)
        print(f"Estimated Fooling Rate: {rate}")

    return v
