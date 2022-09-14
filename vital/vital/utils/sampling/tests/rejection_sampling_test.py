import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from vital.utils.logging import configure_logging
from vital.utils.sampling.rejection_sampling import RejectionSampler


def rejection_sampling_test():
    """Run the test."""
    configure_logging(log_to_console=True, console_level=logging.DEBUG)

    """Tests the implementation of the rejection sampling algorithm."""
    # Initialize target distributions
    normal1 = multivariate_normal(mean=[4, 0], cov=1)
    normal2 = multivariate_normal(mean=[-2, 0], cov=2)

    # Sample from target distributions
    normal1_samples = normal1.rvs(size=250)
    normal2_samples = normal2.rvs(size=250)
    x = np.vstack((normal1_samples, normal2_samples))

    # Sample using the rejection sampling algorithm
    rejection_sampler = RejectionSampler(x)
    samples = rejection_sampler.sample(num_samples=5000)

    # Plot the results
    rcParams["lines.markersize"] /= 3
    plt.figure(dpi=600)
    x, y = np.mgrid[-10:10:0.01, -8:8:0.01]
    pos = np.dstack((x, y))
    plt.contour(x, y, normal1.pdf(pos), colors="red")
    plt.contour(x, y, normal2.pdf(pos), colors="blue")
    plt.scatter(samples[:, 0], samples[:, 1])

    plt.savefig("rejection_sampling_test.png", dpi=600)

    # Compare the target distributions to the gaussian mixture estimated from the samples
    print("\n")
    print("Target distributions:")
    for counter, normal in enumerate([normal1, normal2]):
        print(f"mean_{counter}: {normal.mean} \ncov_{counter}: {normal.cov}")

    mixture = GaussianMixture(n_components=2, covariance_type="spherical").fit(samples)

    print("\n")
    print("Estimated distributions from gaussian mixture:")
    for counter, (mean, cov) in enumerate(zip(mixture.means_, mixture.covariances_)):
        print(f"mean_{counter}: {mean} \ncov_{counter}: {cov}")


if __name__ == "__main__":
    rejection_sampling_test()
