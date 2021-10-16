import sys
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import trange
from sklearn import metrics


class TSNE:
    """ MY realization from Nikita Volkov's applied stats course <3 """

    def __init__(self, iters=1000, n=2):
        self.iters = iters
        self.n = n

    @staticmethod
    def __calc_probs(square_dists, ind, sigma):
        numenator = torch.exp(-square_dists[ind] / (2 * sigma * sigma))
        numenator[ind] = 0
        return numenator / numenator.sum()

    @staticmethod
    def __calc_q_ij(Y):
        square_distances = torch.cdist(Y, Y).pow(2)

        # Assuming Cauchy distribution
        q_ij = 1 / (1 + square_distances)  # not equal but proportional

        # q_ii == 0
        ind = np.diag_indices(q_ij.shape[0])
        q_ij[ind[0], ind[1]] *= 0

        # q_ij == q_ji
        q_ij = q_ij[np.triu_indices(n=q_ij.shape[0])]
        q_ij = q_ij / q_ij.sum()

        return q_ij

    @staticmethod
    def __calc_kl_div(P, Q):
        mask = torch.isclose(P, torch.as_tensor(0.0))  # remove zeros for KL stability :))
        return (P[~mask] * torch.log(P[~mask] / Q[~mask])).sum()

    def __call__(self, X):
        square_distances = torch.cdist(X, X).pow(2)

        # I think it is not very good initialization :( But on my examples it works
        # For stable results this should be random an we should do several tries
        sigmas = [1.0 for _ in range(len(X))]

        # Calculate probabilities from pairwise distances
        conditional_probs = torch.stack([
            TSNE.__calc_probs(square_distances, ind=i, sigma=sigma) for i, sigma in enumerate(sigmas)
        ])

        # p_ij == p_ji
        target_joint_probs = torch.as_tensor(
            conditional_probs[np.triu_indices(n=conditional_probs.shape[0])], dtype=torch.float32
        )

        Y = torch.randn(X.shape[0], self.n, requires_grad=True, dtype=torch.float32)

        opt = torch.optim.Adam(params=[Y], lr=1.0)

        # minimize KL between two distributions
        for _ in trange(self.iters):
            loss = TSNE.__calc_kl_div(target_joint_probs, TSNE.__calc_q_ij(Y))
            opt.zero_grad()
            loss.backward()
            opt.step()

        return Y.detach().numpy()


if __name__ == '__main__':
    # Generate 3 clusters, each consists of 100 points in R^20
    X = torch.cat([torch.rand(100, 20) + 10, torch.rand(100, 20) + 0, torch.rand(100, 20) - 10], axis=0)

    projected = TSNE()(X)

    plt.figure(figsize=(8, 8))
    plt.scatter(*projected.T, alpha=0.5)
    plt.savefig('tsne.jpg')
