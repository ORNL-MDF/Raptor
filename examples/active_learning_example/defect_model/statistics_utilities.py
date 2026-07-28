import numpy as np
import scipy.stats as st
from pathlib import Path


def estimate_lognormal_direct(norm_defect):
    "Approach 1: directly estimate the parameters using standard formulas bases on log transform"
    log_norm_defect = np.log(norm_defect)
    log_mean_defect = np.mean(log_norm_defect)
    log_std_defect = np.std(log_norm_defect, ddof=1)
    log_sem_defect = np.sqrt(1.0 / len(norm_defect)) * log_std_defect

    # formula to estimate variance of the sample variance, requires estimate of fourth moment
    def var_of_sample_var():
        n = len(norm_defect)
        coeff_n = (n / (n - 1)) * (n / (n - 2)) * (n / (n - 3))
        moment4 = coeff_n * np.mean((log_norm_defect - log_mean_defect) ** 4)
        res = (moment4 - (n - 3) / (n - 1) * log_std_defect**4) / n
        return res

    # standard error of the variance (sev)
    log_sev_defect = np.sqrt(var_of_sample_var())

    # this simpler formula is only correct when log(norm_defect) is exactly normally distributed
    # log_sev_defect = np.sqrt(2.0 / (len(norm_defect) - 1)) * log_std_defect**2

    return (log_mean_defect, log_sem_defect), (log_std_defect, log_sev_defect)


def estimate_lognormal_MCMC(norm_defect):
    # Approach 2: using a lightweight mcmc approach assuming a lognormal underlying distribution
    # target y - E[µ] in posterior, yerr - sqrt(Var[µ]) in posterior
    trace = run_metropolis_hastings(
        norm_defect,
        iterations=5000,
        proposal_widths=np.array([1, 1]),
    )
    burnin = 1000
    trace = trace.T[:, burnin:]  # discard burn-in samples

    # extract statistics from MCMC trace
    log_mean_pore = np.mean(trace[0])
    log_sem_pore = np.std(trace[0], ddof=1)
    # square root of the mean variance
    log_std_pore = np.sqrt(np.mean(trace[1] ** 2))
    # standard deviation of the variance
    log_sev_pore = np.std(trace[1] ** 2, ddof=1)

    return (log_mean_pore, log_sem_pore), (log_std_pore, log_sev_pore)


def log_prior_lognormal(params):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf  # log(0)
    mu_prior = st.norm.logpdf(mu, loc=0, scale=10)  # Example prior for mean
    sigma_prior = st.norm.logpdf(sigma, loc=1, scale=5)  # Example prior for std
    return mu_prior + sigma_prior


def loglikelihood_lognormal(params, data):
    mu, sigma = params
    if sigma <= 0:
        return -np.inf  # log(0)
    return np.sum(st.lognorm.logpdf(data, s=sigma, scale=np.exp(mu)))


def log_posterior_lognormal(params, data):
    return loglikelihood_lognormal(params, data) + log_prior_lognormal(params)


def run_metropolis_hastings(
    data, iterations=10000, proposal_widths=np.array([1.0, 2.0])
):
    # Initial guesses
    current_params = np.array([1, 1])  # Example initial guess
    current_log_post = log_posterior_lognormal(current_params, data)

    trace = []

    for i in range(iterations):
        # Propose new parameters (Random Walk)
        proposal = current_params + np.random.normal(
            0, proposal_widths, size=current_params.shape
        )

        proposal_log_post = log_posterior_lognormal(proposal, data)

        # Acceptance ratio
        ratio = np.exp((proposal_log_post - current_log_post))

        if np.random.rand() < ratio:
            current_params = proposal
            current_log_post = proposal_log_post

        trace.append(current_params)

    return np.array(trace)


def estimate_cvar(defects_list, level=0.05):
    "Estimate the conditional value at risk from a finite sample."
    n_defects = len(defects_list)
    n_bad_defects = n_defects * level
    remainder = n_bad_defects - np.floor(n_bad_defects)
    n_bad_defects = int(np.floor(n_bad_defects))
    weights = np.concatenate(([remainder], np.ones(n_bad_defects)))
    weights /= np.sum(weights)
    defects_sort = np.sort(np.asarray(defects_list))
    cvar = np.sum(weights * defects_sort[(-n_bad_defects - 1) :])
    return cvar


def generate_lognormal_defects(lognorm_params, n_defects):
    (log_mean, log_sem), (log_std, log_sev) = lognorm_params
    while True:
        # sample a big defect sample from a random realization of the estimated density
        # find a random mean of the log-normal density model, using the estimated parameters
        log_mu = log_mean + log_sem * np.random.randn(1)
        # find a random variance of the log-normal density model, using the estimated parameters
        log_var = log_std**2
        # this naive formula can produce negative variances, use a relative perturbation
        #  log_s2_old = log_var + log_sev * np.random.randn(1)
        rel_log_sev = log_sev / log_var
        log_s2 = np.exp(np.log(log_var) + rel_log_sev * np.random.randn(1))
        # generate some samples from the uncertain lognormal distribution
        log_s = np.sqrt(log_s2)
        log_samples = log_mu + log_s * np.random.randn(n_defects)
        yield np.exp(log_samples)


def estimate_lognormal_cvar(
    lognorm_params, cvar_level=0.2, n_defects=1000, max_bootstrap=1000
):
    cvar_array = np.zeros((max_bootstrap, 1))
    for n_bs, sample in enumerate(
        generate_lognormal_defects(lognorm_params, n_defects)
    ):
        cvar = estimate_cvar(sample, cvar_level)
        cvar_array[n_bs] = cvar
        if n_bs > 1:
            mean_cvar = np.mean(cvar_array[:n_bs])
            std_cvar = np.std(cvar_array[:n_bs], ddof=1)
        if n_bs + 1 >= max_bootstrap:
            return mean_cvar, std_cvar


def plot_defect_distribution(
    output_filename, sort_defect, normal_parameters, lognorm_parameters, cvar_parameters
):

    (log_mean, log_sem), (log_std, log_sev) = lognorm_parameters
    (mean_pore, std_pore) = normal_parameters
    (mean_cvar, err_cvar) = cvar_parameters
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    defect_mesh = np.linspace(0.01, (mean_pore + 3 * std_pore) * 1e6, 500)
    ax.plot(
        defect_mesh,
        st.norm.pdf(defect_mesh, loc=mean_pore * 1e6, scale=std_pore * 1e6),
        color="tab:blue",
        linewidth=2,
        label="Standard Gaussian estimate",
    )
    ax.plot(
        defect_mesh,
        st.norm.pdf(np.log(defect_mesh), loc=log_mean, scale=log_std) / defect_mesh,
        color="tab:green",
        linewidth=2,
        label="Log Gaussian estimate",
    )
    ax.plot(
        defect_mesh,
        st.gaussian_kde(sort_defect)(defect_mesh),
        color="tab:orange",
        linewidth=2,
        label="Gaussian KDE",
    )
    ax.scatter(
        sort_defect,
        np.zeros(sort_defect.shape),
        color="black",
        marker="+",
        s=15,
        alpha=0.6,
        label="Pore size data in $\\mu$m",
    )
    ax.axvline(mean_cvar * 1e6, color="k", linestyle="-", label="CVAR")
    ax.axvline((mean_cvar + err_cvar) * 1e6, color="k", linestyle=":", label="CVAR+")
    ax.axvline((mean_cvar - err_cvar) * 1e6, color="k", linestyle=":", label="CVAR-")
    ax.axvline(np.max(sort_defect), color="b", linestyle="-", label="maximum")
    ax.legend()
    ax.set_xlabel("defect size")
    ax.set_ylabel("probability density")
    ax.set_title(output_filename)
    plt.tight_layout()
    output_path = Path("pore_plots")
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / output_filename, dpi=300)
    plt.close()
