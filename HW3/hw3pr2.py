import numpy as np
import emcee
import matplotlib.pyplot as plt

# sampling a multi-dimensional Gaussian
# returns the desnity p(x) for specific values of x, mu, and cov
# x is the position of a single walker (a N dimensional numpy array)
def log_prob(x, mu, cov):
    diff = x - mu
    return -0.5 * np.dot(diff, np.linalg.solve(cov, diff))

# Hyperparameters in 5 dimensions
ndim = 5

np.random.seed(42)
means = np.random.rand(ndim)

cov = 0.5 - np.random.rand(ndim ** 2).reshape((ndim, ndim))
cov = np.triu(cov)
cov += cov.T - np.diag(cov.diagonal())
cov = np.dot(cov, cov)

# initlaize starting point for each of the 32 walkers
nwalkers = 32
p0 = np.random.rand(nwalkers, ndim)

# create ensembesamples object
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[means, cov])

# run 100 burn-in steps to let the wlakers explore the parameter spacea bit
# and get settled into the maximum of the density
state = sampler.run_mcmc(p0, 100)

# clears all of the important bookkeeping paramters in the sampler to get a frsh start
# also clears the current position of the walker
sampler.reset()

# do production run of 10000 steps
sampler.run_mcmc(state, 10000)

# access the samples and plot
# the samples would be an array with shape (1000, 32, 5)
samples = sampler.get_chain(flat=True)
plt.hist(samples[:, 0], 100, color="k", histtype="step")
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$p(\theta_1)$")
plt.gca().set_yticks([])
plt.show()

# check the sampling quality with the mean acceptance fraction
# print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

# another way to check is with the integrated autocorrelation time
# print(
#     "Mean autocorrelation time: {0:.3f} steps".format(
#         np.mean(sampler.get_autocorr_time())
#     )
# )