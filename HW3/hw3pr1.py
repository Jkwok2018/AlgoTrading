import matplotlib.pyplot as plt 
import numpy as np
import pymc3
import scipy.stats as stats


plt.style.use("ggplot")
# set the number of coin flip trials carried outa
n = 50
# number of head returned in the trials
z = 10
# prior parameters for the beta distribution
alpha = 12
beta = 12
# posterior parameter, later used to check result from MCMC
alpha_post = 22
beta_post = 52
# iteration of the Metropolis
iterations = 10000

# Use PyMC3 to construct a model context
basic_model = pymc3.Model() 
with basic_model:
    # Define our prior belief about the fairness
    # of the coin using a Beta distribution
    theta = pymc3.Beta("theta", alpha=alpha, beta=beta)
    # Define the Bernoulli likelihood function
    y = pymc3.Binomial("y", n=n, p=theta, observed=z)
    # Carry out the MCMC analysis using the Metropolis algorithm
    # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC 
    start = pymc3.find_MAP()
    # Use the Metropolis algorithm (as opposed to NUTS or HMC, etc.)
    step = pymc3.Metropolis()
    # Calculate the trace
    trace = pymc3.sample(
        iterations, step, start, random_seed=1, progressbar=True
    )

# Plot the posterior histogram from MCMC analysis
bins=50 
plt.hist(
    trace["theta"], bins,
    histtype="step", normed=True,
    label="Posterior (MCMC)", color="red"
)
# Plot the analytic prior and posterior beta distributions
x = np.linspace(0, 1, 100) 
plt.plot(x, stats.beta.pdf(x, alpha, beta),
        "--", label="Prior", color="blue" )

plt.plot(
    x, stats.beta.pdf(x, alpha_post, beta_post), 
    label='Posterior (Analytic)', color="green"
)
# Update the graph labels
plt.legend(title="Parameters", loc="best") 
plt.xlabel("$\\theta$, Fairness") 
plt.ylabel("Density")

 # Show the trace plot
 # trace: vector of samples produced by the MCMC sampling procedure
 # KDE on the left, sampling series on the right
pymc3.traceplot(trace)
plt.show()