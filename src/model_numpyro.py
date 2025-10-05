import numpy as np
from typing import Dict, Any
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp
import jax.random as random
import arviz as az


def bayes_linreg_model(X, y=None):
    """
    Bayesian linear regression with diffuse priors.
    y ~ Normal(alpha + X @ beta, sigma)
    """
    n, d = X.shape
    alpha = numpyro.sample("alpha", dist.Normal(0.0, 10.0))
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(d), 5.0 * jnp.ones(d)))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(5.0))

    mu = alpha + jnp.dot(X, beta)
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def fit_posterior(X_train, y_train, seed: int = 123, chains: int = 4, draws: int = 1000, tune: int = 1000):
    # Avoid the "not enough devices" warning on CPU:
    # set a virtual device count and request sequential chaining.
    numpyro.set_host_device_count(max(1, chains))
    nuts = NUTS(bayes_linreg_model, target_accept_prob=0.9)
    mcmc = MCMC(
        nuts,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",  # be explicit on CPU
        progress_bar=False,
    )
    rng_key = random.PRNGKey(seed)
    mcmc.run(rng_key, X=X_train, y=y_train)
    mcmc.print_summary(exclude_deterministic=False)
    return mcmc


def to_inferencedata(mcmc) -> az.InferenceData:
    return az.from_numpyro(mcmc)


def posterior_predict(idata: az.InferenceData, X: np.ndarray, seed: int = 123, draws: int = 1000) -> np.ndarray:
    """
    Return predictive samples [S, n] drawing from posterior.
    """
    posterior = idata.posterior
    # Stack chain, draw
    alpha = posterior["alpha"].stack(sample=("chain", "draw")).values  # [S]
    beta = posterior["beta"].stack(sample=("chain", "draw")).values  # [S, d] or [d, S] depending
    if beta.shape[0] != alpha.shape[0]:  # ensure [S, d]
        beta = np.swapaxes(beta, 0, 1)
    sigma = posterior["sigma"].stack(sample=("chain", "draw")).values  # [S]

    S = min(draws, alpha.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(alpha.shape[0], size=S, replace=False)
    a = alpha[idx]  # [S]
    b = beta[idx]  # [S, d]
    s = sigma[idx]  # [S]

    mu = X @ b.T  # [n, S]
    mu = mu + a[None, :]  # broadcast add intercept across columns
    eps = rng.normal(0.0, 1.0, size=mu.shape) * s[None, :]  # [n, S]
    y_samp = mu + eps
    return y_samp.T  # [S, n]
