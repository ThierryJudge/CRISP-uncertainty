import logging
from typing import Literal, Tuple

import numpy as np
from numpy.random import SeedSequence
from pathos.multiprocessing import Pool
from scipy.stats import multivariate_normal
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

logger = logging.getLogger(__name__)


class RejectionSampler:
    """Generic implementation of the rejection sampling algorithm."""

    def __init__(
        self,
        data: np.ndarray,
        kde_bandwidth: float = None,
        proposal_distribution_params: Tuple[np.ndarray, np.ndarray] = None,
        scaling_mode: Literal["max", "3rd_quartile"] = "max",
    ):
        """Initializes the inner distributions used by the rejection sampling algorithm.

        Args:
            data: N x D array where N is the number of data points and D is the dimensionality of the data.
            kde_bandwidth: Bandwidth of the kernel density estimator. If no bandwidth is given, it will be determined
                by cross-validation over ``data``.
            proposal_distribution_params: `mean` and `cov` parameters to use for the Gaussian proposal distribution. If
                no params are given, the proposal distribution is inferred from the mean and covariance computed on
                ``data``.
            scaling_mode: Algorithm to use to compute the scaling factor between the proposal distribution and the KDE
                estimation of the real distribution.
        """
        self.data = data

        # Init kernel density estimate
        if kde_bandwidth:
            self.kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian").fit(self.data)
        else:
            logger.info("Cross-validating bandwidth of kernel density estimate...")
            grid = GridSearchCV(
                KernelDensity(kernel="gaussian"), {"bandwidth": 10 ** np.linspace(-1, 1, 100)}, cv=ShuffleSplit()
            )
            grid.fit(self.data)
            self.kde = grid.best_estimator_
            logger.info(f"Parameters of KDE optimized for the data: {self.kde}.")

        # Init proposal distribution
        if proposal_distribution_params:
            mean, cov = proposal_distribution_params
        else:
            mean = np.mean(self.data, axis=0)
            cov = np.cov(self.data, rowvar=False)
        self.proposal_distribution = multivariate_normal(mean=mean, cov=cov)

        # Init scaling factor
        factors_between_data_and_proposal_distribution = (
            np.e ** self.kde.score_samples(self.data)
        ) / self.proposal_distribution.pdf(self.data)
        if scaling_mode == "max":
            # 'max' scaling mode is used when the initial samples fit in a sensible distribution
            # Should be preferred to other algorithms whenever it is applicable
            self.scaling_factor = np.max(factors_between_data_and_proposal_distribution)
        else:  # scaling_mode == '3rd_quartile'
            # '3rd_quartile' scaling mode is used when outliers in the initial samples skew the ratio and cause an
            # impossibly high scaling factor
            self.scaling_factor = np.percentile(factors_between_data_and_proposal_distribution, 75)

    def sample(self, num_samples: int, batch_size: int = None) -> np.ndarray:
        """Performs rejection sampling to sample M samples that fit the visible distribution of ``data``.

        Args:
            num_samples: Number of samples to sample from the data distribution.
            batch_size: Number of samples to generate in each batch. If ``None``, defaults to ``num_samples / 100``.

        Returns:
            M x D array where M equals `num_samples` and D is the dimensionality of the sampled data.
        """
        # Determine the size and number of batches (possibly including a final irregular batch)
        if batch_size is None:
            batch_size = num_samples // 100
            logger.info(f"No `batch_size` provided. Defaulted to use a `batch_size` of {batch_size}.")
        batches = [batch_size] * (num_samples // batch_size)
        if last_batch := num_samples % batch_size:
            batches.append(last_batch)

        # Prepare different seeds for each batch
        ss = SeedSequence()
        logger.info(f"Entropy of root `SeedSequence` used to spawn generators: {ss.entropy}.")
        rngs = [np.random.default_rng(seed) for seed in ss.spawn(len(batches))]

        # Sample batches in parallel using a pool of processes
        with Pool() as pool:
            sampling_result = tqdm(
                pool.imap(lambda args: self._sample(*args), zip(batches, rngs)),
                total=len(batches),
                desc="Sampling from observed data distribution with rejection sampling",
                unit="batch",
            )
            samples, nb_trials = zip(*sampling_result)

        samples = np.vstack(samples)  # Merge batches of samples in a single array
        nb_trials = sum(nb_trials)  # Sum over the number of points sampled to get each batch

        # Log useful information to analyze/debug the performance of the rejection sampling
        logger.debug(f"Number of unique samples generated: {len(np.unique(samples, axis=0))}")
        logger.info(
            "Percentage of generated samples accepted by rejection sampling: "
            f"{round(samples.shape[0] / nb_trials * 100, 2)} \n"
        )

        return samples

    def _sample(self, num_samples: int, rng: np.random.Generator = None) -> Tuple[np.ndarray, int]:
        """Performs rejection sampling to sample M samples that fit the visible distribution of ``data``.

        `self._sample` performs the sampling in itself, as opposed to `self.sample` which is a public wrapper to
        coordinate sampling multiple batches in parallel.

        Args:
            num_samples: Number of samples to sample from the data distribution.
            rng: Random Number Generator to use to draw from both the proposal and uniform distributions.

        Returns:
            samples: M x D array where M equals `num_samples` and D is the dimensionality of the sampled data.
            nb_trials: Number of draws (rejected or accepted) it took to reach M accepted samples. This is mainly useful
                to evaluate the efficiency of the rejection sampling.
        """
        if rng is None:
            rng = np.random.default_rng()
        samples = []
        nb_trials = 0
        while len(samples) < num_samples:
            sample = self.proposal_distribution.rvs(size=1, random_state=rng)
            rand_likelihood_threshold = rng.uniform(0, self.scaling_factor * self.proposal_distribution.pdf(sample))

            if rand_likelihood_threshold <= (np.e ** self.kde.score_samples(sample[np.newaxis, :])):
                samples.append(sample)

            nb_trials += 1

        return np.array(samples), nb_trials
