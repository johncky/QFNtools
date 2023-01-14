import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pykalman import KalmanFilter
from scipy.stats import entropy
from statsmodels.sandbox.tools.tools_pca import pca
import requests
from io import StringIO
from bs4 import BeautifulSoup
from scipy.optimize import minimize
from scipy.stats import norm
import pandas_market_calendars as mcal
from time import strptime
from numpy.linalg import inv
from scipy.special import loggamma
import scipy
from scipy.sparse import diags

class EfficientFrontier:
    def __init__(self, risk_measure, alpha=5, entropy_bins=None):
        self.risk_measure = risk_measure.lower()
        self.df = None

        # percentile for VaR
        self.alpha = alpha
        self.entropy_bins = entropy_bins

        # covariance matrix
        self.omega = None

        # mean vector
        self.R = None

        self.wgt = None
        self.mu_range = None
        self.risk_range = None

    def fit(self, df, wbnd, mu_range):
        self.R = df.mean()
        self.omega = df.cov()
        self.df = df

        if self.risk_measure == 'cvar':
            obj_func = self.cvar
        elif self.risk_measure == 'var':
            obj_func = self.var
        elif self.risk_measure == 'entropy':
            obj_func = self.entropy_
        else:
            obj_func = self.sd

        risk_range = np.zeros(len(mu_range))
        n = df.shape[1]
        wgt = list()

        for i in range(len(mu_range)):
            mu = mu_range[i]

            # initial weight = equal weight
            x_0 = np.ones(n) / n

            # bounds for weightings
            bndsa = [wbnd for j in range(n)]

            # constraint 1 --> type=equality --> sum(weightings) = 1
            # constraint 2 --> type=equality --> np.dot(w^T, R) = mu
            consTR = (
                {'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}, {'type': 'eq', 'fun': lambda x: mu - np.dot(x, self.R)})

            # Find min risk portfolio for given mu
            w = minimize(obj_func, x_0, method='SLSQP', constraints=consTR, bounds=bndsa)

            risk_range[i] = obj_func(w.x)

            wgt.append(np.squeeze(w.x))

        wgt = np.array(wgt)
        self.wgt = wgt
        self.mu_range = mu_range
        self.risk_range = risk_range

    def sd(self, w):
        return np.dot(w, np.dot(self.omega, w.T))

    def cvar(self, w):
        ret = np.dot(self.df, w.T)
        return abs(min((np.mean(ret[ret <= np.percentile(ret, self.alpha)]), 0)))

    def var(self, w):
        ret = np.dot(self.df, w.T)
        return abs(min(np.percentile(ret, self.alpha), 0))

    def entropy_(self, w):
        ret = np.dot(self.df, w.T)
        ret = pd.Series(ret)
        if self.entropy_bins is None:
            bins = int(np.sqrt(ret.shape[0] / 5))
        else:
            bins = self.entropy_bins
        counts = ret.value_counts(bins=bins)
        h = entropy(counts)
        return h

    def weights(self, drop_zero_col=True, rounding=True):
        df = pd.DataFrame(self.wgt, index=[self.mu_range, self.risk_range], columns=self.df.columns)
        df.index.names = ['mu', self.risk_measure]
        if rounding:
            df = np.round(df, 2)
        if drop_zero_col:
            df = df.loc[:, (df != 0.0).any(axis=0)]
        return df

    def to_csv(self, path):
        self.weights().to_csv(path)

    def plot(self, port_only=True):
        risk_range = self.risk_range
        mu_range = self.mu_range
        if self.risk_measure == 'cvar':
            label = '{}% Conditional VaR (%)'.format(self.alpha)
        elif self.risk_measure == 'var':
            label = '{}% VaR (%)'.format(self.alpha)
        elif self.risk_measure == 'entropy':
            label = 'Entropy'
        else:
            risk_range = np.sqrt(risk_range)
            label = 'Standard Deviation (%)'
        if port_only:
            wgt = self.weights()
            idx = (wgt != 0).sum(1) != 1
            risk_range = risk_range[idx]
            mu_range = mu_range[idx]
        plt.plot(np.multiply(risk_range, 100), np.multiply(mu_range, 100), color="red")
        plt.xlabel(label, fontsize=10)
        plt.ylabel("Expected Return (%)", fontsize=10)
        plt.title("Efficient Frontier", fontsize=12)


class DynamicBeta:
    def __init__(self):
        self.kf = None
        self.filter_df = None
        self.smoothed_df = None
        self.x = None
        self.y = None

    def fit(self, y, x, factor_pca=False, n_pc=3):
        if type(y) == pd.Series:
            y = pd.DataFrame(y)
        if type(x) == pd.Series:
            x = pd.DataFrame(x)

        if factor_pca:
            xreduced, factors, evals, evecs = pca(x, keepdim=n_pc)
            x = pd.DataFrame(factors, index=x.index, columns=['PC{}'.format(i) for i in range(1, n_pc + 1)])

        n_dim_obs = y.shape[1]
        n_dim_state = x.shape[1] + 1
        ntimestep = y.shape[0]
        factors = sm.add_constant(x)

        fac_obs = np.array(factors)
        obs_matrics = np.zeros((ntimestep, n_dim_obs, n_dim_state))

        for i in range(n_dim_obs):
            obs_matrics[:, i, :] = fac_obs

        kf = KalmanFilter(n_dim_obs=n_dim_obs, n_dim_state=n_dim_state, transition_matrices=np.eye(factors.shape[1]),
                          observation_matrices=obs_matrics,
                          em_vars=['transition_covariance', 'observation_covariance', 'initial_state_mean',
                                   'initial_state_covariance'])

        if factor_pca:
            cols = ['Intercept'] + ['beta-PC{}'.format(i) for i in range(1, n_pc + 1)]
        else:
            cols = ['Intercept'] + list(x.columns)

        self.kf = kf
        filter_state_means, filter_state_covs = kf.filter(y)

        self.filter_df = pd.DataFrame(filter_state_means, index=y.index, columns=cols)
        smoothed_state_means, smoothed_state_covs = kf.smooth(y)
        self.smoothed_df = pd.DataFrame(smoothed_state_means, index=y.index, columns=cols)

    def plot(self, smoothed=False):
        if smoothed:
            self.smoothed_df.plot()
        else:
            self.filter_df.plot()


# Diagnostic plots for MCMC samples
class Diagnostic:
    def __init__(self, samples):
        self.samples = samples

    # draw autocorrelation plots to identify stickiness of MCMC chain
    def acf(self, max_var=5, max_lags=100):
        samples = getattr(self, 'samples')
        var_names = samples.columns

        n_var = min(max_var, samples.shape[1])
        fig, axes = plt.subplots(nrows=n_var, ncols=1, sharex=True)
        max_lags = min(max_lags, samples.shape[0])
        for i in range(n_var):
            ax = axes[i]
            acorr = sm.tsa.acf(samples.iloc[:, i], nlags=max_lags)
            ax.bar(range(0, max_lags + 1), acorr)
            ax.set_title(var_names[i], fontsize=10)
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        fig.suptitle('Autocorrelation')
        fig.supxlabel('lags')
        fig.supylabel('corr')
        plt.show()

    # draw traceplot to ensure convergence & good mixing
    def traceplot(self, max_var=5, max_n=1000):
        samples = getattr(self, 'samples')
        var_names = samples.columns

        n_var = min(max_var, samples.shape[1])
        fig, axes = plt.subplots(nrows=n_var, ncols=1, sharex=True)
        max_n = min(max_n, samples.shape[0])
        for i in range(n_var):
            ax = axes[i]
            ax.plot(range(1, max_n + 1), samples.iloc[:max_n, i])
            ax.set_title(var_names[i], fontsize=10)
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.97])
        fig.suptitle('Traceplot')
        fig.supxlabel('id')
        fig.supylabel('val')
        plt.show()

    # draw posterior density of all variables
    def density_plot(self):
        samples = getattr(self, 'samples')
        p = sns.pairplot(samples, diag_kind="kde", plot_kws=dict(alpha=1,
                                                                 hue=samples.index,
                                                                 palette="blend:gold,dodgerblue"))
        p.map_lower(sns.kdeplot, color=".2")
        p.fig.suptitle('Posterior Density')
        p.fig.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()

    # compute posterior mean of variables
    def mean(self):
        return self.samples.mean()

    # compute posterior credible interval of variables
    def credible_interval(self, alpha=0.05):
        samples = getattr(self, 'samples')
        return samples.quantile([alpha / 2, 1 - alpha / 2])

    # approximation of effective sample size using autocorrelation
    def ess(self):
        samples = getattr(self, 'samples')
        return samples.shape[0] / (samples.apply(func=lambda x: sm.tsa.acf(x)[1:]).sum() * 2 + 1)

    # E[Y|X] = int E[Y|X,beta] p(beta|X) dbeta
    # compute the conditional mean of Y given new X & data
    def predict(self, X):
        samples = getattr(self, 'samples')
        return X.dot(samples.iloc[:, :-1].T)

    # generate samples of Y given new X, using beta samples from MCMC
    # noise is added here, assuming that the noises are iid normal with variance = posterior sigma2 samples
    def predictive_dist(self, X):
        samples = getattr(self, 'samples')
        S = samples.shape[0]
        noise = np.random.normal(0, np.sqrt(samples.iloc[:, -1]), size=[X.shape[0], S])
        y_samples = X.dot(samples.iloc[:, :-1].T) + noise

        return y_samples

    # compute the predictive confidence interval of Y given a new X, using samples from MCMC
    def predictive_ci(self, X, q=(0.025, 0.975)):
        y_samples = self.predictive_dist(X)
        return np.quantile(y_samples, q, axis=1).T

    # plot the predictive posterior checks
    # compare the simulated dataset generated from our model to the actual dataset
    # a good model should generate simulated datasets close to actual dataset
    def ppc_plot(self, X, y, max_sample=100, **kwargs):
        samples = getattr(self, 'samples')
        sim_y = X.dot(samples.iloc[:, :-1].T) + np.random.normal(0, np.sqrt(samples.iloc[:, -1]))
        for i in range(min(sim_y.shape[1], max_sample)):
            sns.kdeplot(data=sim_y[:, i], alpha=0.1)
        sns.kdeplot(data=y, color='black', linewidth=1, label='data', **kwargs)
        sns.kdeplot(data=np.mean(sim_y, axis=1), color='red', linewidth=1, label='Simulated mean', **kwargs)

        ols_y = X.dot(inv(X.T.dot(X)).dot(X.T).dot(y))
        sns.kdeplot(data=ols_y, color='blue', linewidth=1, label='OLS', **kwargs)

        plt.legend()
        plt.show()


# Implementation of bayesian linear regression with Normal beta prior, Inverse-gamma noise sigma2
class BayesLinReg:
    def __init__(self):
        self.res = np.nan

    # breakdown formula and extract X & y from DataFrame
    def fit(self, data, formula, beta0, lambda0, sigma02, v0):
        str_seg = formula.split('~')
        resp = str_seg[0].strip()
        if str_seg[1].strip() == '.':
            predictors = list(data.columns)
            predictors.remove(resp)
        else:
            predictors = [v.strip() for v in str_seg[1].split('+')]

        X = data[predictors].to_numpy()
        y = data[resp].to_numpy()
        coef_names = ['beta_{}'.format(var) for var in predictors] + ['sigma2']

        setattr(self, 'formula', formula)
        setattr(self, 'predictors', predictors)
        setattr(self, 'response', resp)
        setattr(self, 'coef_names', coef_names)

        self._fit(X, y, beta0, lambda0, sigma02, v0)

    # compute the stuff necessary for drawing MCMC samples
    def _fit(self, X, y, beta0, lambda0, sigma02, v0):
        XTX = X.T.dot(X)
        lambda0_inv = inv(lambda0)
        beta_ols = inv(XTX).dot(X.T).dot(y)
        n = X.shape[0]
        p = beta0.shape[0]

        _local_vars = locals()
        _local_var_names = list(_local_vars.keys())
        _local_var_names.remove('self')
        for var in _local_var_names:
            setattr(self, var, _local_vars[var])

    # function to draw from full conditionals of beta
    def beta_full_conditionals(self, XTX, lambda0_inv, beta_ols, beta0, sigma2):

        sample_precision = XTX / sigma2
        lambdan_inv = sample_precision + lambda0_inv
        lambdan = inv(lambdan_inv)
        beta_n = lambdan.dot(sample_precision.dot(beta_ols) + lambda0_inv.dot(beta0))

        return np.random.multivariate_normal(mean=beta_n, cov=lambdan, size=1).flatten()

    # function to draw from full conditionals of sigma2
    def sigma2_full_conditionals(self, X, y, v0, n, sigma02, beta):

        residual = y - X.dot(beta)
        SSR_beta = residual.T.dot(residual)

        return 1 / np.random.gamma((v0 + n) / 2, 2 / (v0 * sigma02 + SSR_beta), size=1)

    # Gibbs sampling to sample from posterior
    def sample_posterior(self, sample_size=10000, burn_ins=1000):
        XTX = getattr(self, 'XTX')
        X = getattr(self, 'X')
        lambda0_inv = getattr(self, 'lambda0_inv')
        beta_ols = getattr(self, 'beta_ols')
        n = getattr(self, 'n')
        p = getattr(self, 'p')
        y = getattr(self, 'y')
        beta0 = getattr(self, 'beta0')
        sigma02 = getattr(self, 'sigma02')
        v0 = getattr(self, 'v0')

        beta = beta0
        samples = np.zeros(shape=[sample_size, p + 1])

        for i in range(burn_ins):
            sigma2 = self.sigma2_full_conditionals(X, y, v0, n, sigma02, beta)
            beta = self.beta_full_conditionals(XTX, lambda0_inv, beta_ols, beta0, sigma2)

        for i in range(sample_size):
            sigma2 = self.sigma2_full_conditionals(X, y, v0, n, sigma02, beta)
            beta = self.beta_full_conditionals(XTX, lambda0_inv, beta_ols, beta0, sigma2)
            samples[i, :-1] = beta
            samples[i, -1] = sigma2

        samples = pd.DataFrame(samples, columns=self.coef_names)
        self.res = Diagnostic(samples)
        return self.res

# Zellner's g prior
# g is shrinkage parameter, smaller g, posterior of beta is shrink towards 0. larger g, posterior mean converges to OLS
# beta
class ZellnerLinReg(BayesLinReg):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def fit(self, data, formula, sigma02, v0):
        super().fit(data, formula, beta0=np.nan, lambda0=np.nan, sigma02=sigma02, v0=v0)

    # compute stuff requires for drawing posterior samples
    # log marginal likelihood log[p(x|Model)] is also computed for model averaging
    def _fit(self, X, y, beta0, lambda0, sigma02, v0):
        beta0 = np.zeros(len(self.predictors))
        lambda0 = np.identity(1)
        super()._fit(X, y, beta0, lambda0, sigma02, v0)

        g = self.g

        SSR_g = y.T.dot(y - g / (1 + g) * X.dot(self.beta_ols))
        vnsigman2 = v0 * sigma02 + SSR_g
        scale_n = 2 / vnsigman2

        n = self.n
        vn = v0 + n
        log_marginal_likelihood = (-n / 2) * np.log(np.pi) + loggamma(vn / 2) - loggamma(v0 / 2) + (
                    -self.p / 2) * np.log(1 + g) + (v0 / 2) * np.log(v0 * sigma02) - (vn / 2) * np.log(vnsigman2)

        setattr(self, 'log_marginal_likelihood', log_marginal_likelihood)
        setattr(self, 'scale_n', scale_n)

    def beta_full_conditionals(self, XTX, lambda0_inv, beta_ols, beta0, sigma2):
        g = self.g

        sample_precision = XTX / sigma2
        lambdan_inv = sample_precision * (1 + 1 / g)
        lambdan = inv(lambdan_inv)
        beta_n = (g / (1 + g)) * beta_ols
        return np.random.multivariate_normal(mean=beta_n, cov=lambdan, size=1).flatten()

    # it is not full conditionals
    # it is actually posterior distribution, since the priors are not independent in Zellner's g
    def sigma2_full_conditionals(self, X, y, v0, n, sigma02, beta):
        g = self.g
        scale_n = self.scale_n
        return 1 / np.random.gamma((v0 + n) / 2, scale_n, size=1)

    # this is not Gibbs' sampler
    # this is standard Monte Carlo: draw sigma2 from posterior, draw beta | sigma2 from posterior
    # then, we have true joint posterior
    def sample_posterior(self, sample_size=10000):
        return super().sample_posterior(sample_size=10000, burn_ins=0)


# Unit information prior
# it is simply g=n (sample size) in Zellner's g prior
class UnitInfoLinReg(ZellnerLinReg):
    def __init__(self):
        super(ZellnerLinReg, self).__init__()

    @property
    def g(self):
        return self.X.shape[0]

# using Bayesian model averaging (BMA) on top of Zellner's g prior
# one requirement for using BMA is that we know the marginal likelihood of a model: P(X|model)
# which is possible in closed form using Zellner's g prior
class BMA_LinReg:
    def __init__(self, g):
        self.samples = np.nan
        self.g = g
        self.X = np.nan
        self.y = np.nan
        self.predictors = np.nan
        self.resp = np.nan

        self.seen_models = list()
        self.model_params = dict()

    def fit(self, data, y, v0, sigma02):
        data = data.copy()
        y_series = data.pop(y)

        self.resp = y_series.name
        self.predictors = list(data.columns)
        self.X = data.to_numpy()
        self.y = y_series.to_numpy()
        self.v0 = v0
        self.sigma02 = sigma02

    def fit_submodel(self, X, y, z):
        v0 = self.v0
        sigma02 = self.sigma02
        g = self.g

        X_new = X[:, z == 1]
        XTX = X_new.T.dot(X_new)
        beta_ols = inv(XTX).dot(X_new.T).dot(y)
        n = X_new.shape[0]
        p = X_new.shape[1]
        vn = v0 + n
        SSR_g = y.T.dot(y - g / (1 + g) * X_new.dot(beta_ols))
        vnsigman2 = v0 * sigma02 + SSR_g
        log_marginal_likelihood = (-n / 2) * np.log(np.pi) + loggamma(vn / 2) - loggamma(v0 / 2) + (-p / 2) * np.log(
            1 + g) + (v0 / 2) * np.log(v0 * sigma02) - (vn / 2) * np.log(vnsigman2)

        return (vnsigman2, log_marginal_likelihood, beta_ols, XTX)

    def log_marginal_likelihood(self, X, y, z):
        z_str = ''.join([str(int(x)) for x in z])
        if z_str in self.seen_models:
            val = self.model_params[z_str]['log_marginal_likelihood']
        else:
            res = self.fit_submodel(X, y, z)
            val = res[1]

            params = {'vnsigman2': res[0],
                      'log_marginal_likelihood': val,
                      'beta_ols': res[2],
                      'XTX': res[3]}

            self.seen_models.append(z_str)
            self.model_params[z_str] = params
        return val

    def sample_from_model(self, X, y, z, v0, n, g):
        z_str = ''.join([str(int(x)) for x in z])
        if z_str in self.seen_models:
            XTX = self.model_params[z_str]['XTX']
            beta_ols = self.model_params[z_str]['beta_ols']
            vnsigman2 = self.model_params[z_str]['vnsigman2']
        else:
            res = self.fit_submodel(X, y, z)

            XTX = res[3]
            beta_ols = res[2]
            vnsigman2 = res[0]

            params = {'vnsigman2': vnsigman2,
                      'log_marginal_likelihood': res[1],
                      'beta_ols': beta_ols,
                      'XTX': XTX}

            self.seen_models.append(z_str)
            self.model_params[z_str] = params

        sigma2_sample = 1 / np.random.gamma((v0 + n) / 2, 2 / vnsigman2, size=1)

        lambdan = inv(XTX / sigma2_sample * (1 + 1 / g))
        beta_n = (g / (1 + g)) * beta_ols
        beta_sample = np.random.multivariate_normal(mean=beta_n, cov=lambdan, size=1).flatten()

        return (beta_sample, sigma2_sample)

    def sample_posterior(self, sample_size=10000):
        X = self.X
        y = self.y
        g = self.g
        v0 = self.v0

        max_p = X.shape[1]
        n = X.shape[0]
        z = np.ones(max_p)

        samples = np.empty(shape=[sample_size, 2 * max_p + 1])
        samples[:] = np.nan

        for i in range(sample_size):
            for j in range(max_p):
                z_incld = z.copy()
                z_incld[j] = 1

                z_excld = z.copy()
                z_excld[j] = 0

                log_odd_diff = self.log_marginal_likelihood(X, y, z_incld) - self.log_marginal_likelihood(X, y, z_excld)
                if log_odd_diff > 709.78:
                    z[j] = 1
                else:
                    odd = np.exp(
                        self.log_marginal_likelihood(X, y, z_incld) - self.log_marginal_likelihood(X, y, z_excld))
                    z[j] = np.random.binomial(1, odd / (1 + odd))

            if z.sum() == 0:
                continue
            model_sample = self.sample_from_model(X, y, z, v0, n, g)
            samples[i:, 0:max_p] = z
            samples[i:, np.where(z == 1)[0] + max_p] = model_sample[0]
            samples[i:, -1] = model_sample[1]

        predictors = self.predictors
        cols = ['include_{}'.format(x) for x in predictors] + predictors + ['sigma2']
        samples = pd.DataFrame(samples, columns=cols)
        self.samples = samples
        return samples

    def predictive_dist(self, X):
        samples = self.samples
        p = len(self.predictors)
        betas = samples.iloc[:, p:2 * p]
        betas = np.nan_to_num(betas, 0)
        return X.dot(betas.T)

    def predict(self, X):
        return np.mean(self.predictive_dist(X), axis=1)

    def diagnostic(self):
        X = self.X
        y = self.y
        samples = self.samples
        y_pred = self.predict(X)

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, figsize=(11, 11))

        # probability of inclusion
        include_p = samples.iloc[:, :X.shape[1]]
        include_p = include_p.mean().to_numpy()
        ax = axes[0, 0]
        sns.barplot(x=self.predictors, y=include_p, ax=ax, color='b')
        ax.set_xlabel('Predictors')
        ax.set_ylabel('Probability')
        ax.set_title('Probability of inclusion')

        # predicted Y vs Y scatter plot
        ols_y = X.dot(inv(X.T.dot(X)).dot(X.T).dot(y))
        ax = axes[0, 1]
        sns.regplot(x=y, y=y_pred, ax=ax, color='b', ci=None, scatter_kws={'s': 0.5}, line_kws={'linewidth': 0.7},
                    label='BMA')
        sns.regplot(x=y, y=ols_y, ax=ax, color='r', ci=None, scatter_kws={'s': 0.5}, line_kws={'linewidth': 0.7},
                    label='OLS')
        # sns.regplot(x=x, y=y)
        ax.set_xlabel('Y')
        ax.set_ylabel('Y pred')
        ax.set_title('Predicted Y')
        ax.legend()

        # error density
        ols_err = ols_y - y
        bma_err = y_pred - y
        ols_mse = np.round(np.mean(ols_err ** 2), 2)
        bma_mse = np.round(np.mean(bma_err ** 2), 2)
        ax = axes[1, 0]
        sns.kdeplot(data=ols_err, color='r', linewidth=1, label='OLS MSE:{}'.format(ols_mse), ax=ax, alpha=0.5,
                    fill=True)
        sns.kdeplot(data=bma_err, color='b', linewidth=1, label='BMA MSE:{}'.format(bma_mse), ax=ax, alpha=0.5,
                    fill=True)
        ax.legend()
        ax.set_xlabel('Error')
        ax.set_title('Error density')

        # ppc checks
        ax = axes[1, 1]
        sim_y = self.predictive_dist(X)
        for i in range(min(sim_y.shape[1], 500)):
            sns.kdeplot(data=sim_y[:, i], alpha=0.1, ax=ax)
        sns.kdeplot(data=y, color='black', linewidth=1, label='data', ax=ax)
        ax.set_xlabel('value')
        ax.set_ylabel('density')
        ax.set_title('Posterior predictive check')
        ax.legend()

        plt.show()


# scrape HSI option prices & construct risk-neutral density using various methods
class HsiRND:
    def __init__(self, date, maturity_id, option_type='C'):
        self.date = date

        # maturity_id is in the format: YYYYMMDD
        self.maturity_id = maturity_id
        option_type = option_type.upper()
        assert (option_type in ('C', 'P')), 'Option Type must be "C" or "P"'
        self.option_type = option_type

        # download raw option data from HKEX
        self.option_df = self.download_option_data()

        # filter option data to only the selected maturity
        # get the string name of maturity, e.g. MAR-22
        maturity_str = self.option_df.maturity.unique()[
            maturity_id]

        # filtered_df
        self.filtered_df = \
            self.option_df.loc[(self.option_df.maturity == maturity_str) & (self.option_df.type == self.option_type)][
                ['strike', 'IV%', 'Close']]

        # calculate spot HSI S0, maturity T in years (trading days only), risk-free rate rf
        self.T = self.get_T(maturity_str=maturity_str)
        self.rf = self.get_rf()  # 1-year HIBOR is taken as risk-free
        # S0 is approximately option value of extremely OTM option + strike
        if self.option_type == 'C':
            self.S0 = (self.filtered_df['Close'] + self.filtered_df.strike).iloc[0]
        else:
            self.S0 = (self.filtered_df['Close'] + self.filtered_df.strike).iloc[-1]

        # store fitted pdf/pmf
        self.fitted_models = dict()

    # fit using BLA method
    # min_hsi, max_hsi: infer density between [min_hsi, max_hsi]
    def fit_BLA(self, evaluation_steps=5, min_hsi=5, max_hsi=100000):
        filtered_df = self.filtered_df.copy()

        # keep track of changes in strike gap
        k = filtered_df['strike']
        gap_chg = k.diff().diff().fillna(0)

        # identify the index of strike where change in strike gap occurs
        idx_gap_chg = gap_chg.loc[gap_chg != 0.0].index

        # calculate probability mass function from option prices, pmf is also the butterfly spread prices
        price = filtered_df['Close']
        # formula for butterfly spread prices
        bfly_spr = (-2 * price + price.shift(1) + price.shift(-1))

        # Prices of Butterfly spread with these strikes cannot be calculated, as prices of some strikes are not available
        # due to the changing strike gap
        for i in idx_gap_chg:
            bfly_spr.loc[i - 1] = 0

        # divide by strike difference to obtain pmf
        pmf = bfly_spr / k.diff().fillna(1)
        pmf.index = k

        # enforce monotonicity in cmf, remove negative pmf using interpolation
        fixed_pmf = self.enforce_cmf_monotonicity(pmf)
        fixed_pmf = pd.DataFrame(fixed_pmf, columns=['pmf'])

        # divide by normalizing constant so that they sum to 1, this may not be appropriate!
        fixed_pmf = fixed_pmf / fixed_pmf.sum()
        self.fitted_models['BLA'] = fixed_pmf # store fitted pmf
        return fixed_pmf

    # given pd.Series with index=strikes, values=pmf, it remove (-ve) vals and interpolate cmf
    def enforce_cmf_monotonicity(self, density):
        strikes = density.index
        cmf = density.cumsum()
        pos_cmf = cmf.loc[cmf.diff() >= 0]
        pos_cmf = pos_cmf.loc[pos_cmf.diff() >= 0]

        # interpolate & calculate pmf
        fixed_pmf = pos_cmf.reindex(strikes).interpolate(method='pchip').diff().fillna(0)
        return fixed_pmf

    # Use kernel regression to interpolate the implied volatility curve,
    # and obtain "artificial" option prices for a continuum of strikes using BS formulas
    # then use BLA method to compute RND
    # Parameters:
    # wb of kernal regression = wb_mult * 200
    # evaluatuion_steps: the gap between strikes
    # min_hsi, max_hsi: min & max of hsi points at which to evaluate IV
    # min_moneyness, max_moneyness: evaluate RND at [min & max moneyness] * strikes
    def fit_IV(self, wb_mult=4, evaluation_steps=5, min_hsi=5, max_hsi=100000, min_moneyness=0.3, max_moneyness=2):
        filtered_df = self.filtered_df

        # get estimation points from (min_hsi, max_hsi, evaluation_steps)
        est_pts = np.arange(min_hsi, max_hsi, evaluation_steps)

        # run kernel regression on Black-Scholes Implied Volatility
        kernel_iv = self.kernel_regression(x=filtered_df['strike'], y=filtered_df['IV%'].values, est_pts=est_pts,
                                           wb=200 * wb_mult)
        kernel_iv = kernel_iv.fillna(method='ffill').fillna(method='bfill').clip(
            lower=1)  # BS IV cannot be 0, set lower bound to 1

        # convert Black-Scholes IV to option value through BSF, and calculate RND using BLA
        S0 = self.S0
        if self.option_type == 'C':
            prices = pd.Series(self.bs_call(S0, kernel_iv.index.values, self.T, self.rf, (kernel_iv['y'] / 100).values),
                               index=kernel_iv.index)
            BLA_pmf = (-2 * prices + prices.shift(1) + prices.shift(-1)).fillna(0) / evaluation_steps
        else:
            prices = pd.Series(self.bs_put(S0, kernel_iv.index.values, self.T, self.rf, (kernel_iv['y'] / 100).values),
                               index=kernel_iv.index).diff().diff()
            BLA_pmf = (-2 * prices + prices.shift(1) + prices.shift(-1)).fillna(0) / evaluation_steps

        BLA_pmf = self.enforce_cmf_monotonicity(BLA_pmf)
        BLA_pmf = BLA_pmf.loc[BLA_pmf.index >= S0 * min_moneyness]  # set a lower bound range for strike
        BLA_pmf = BLA_pmf.loc[BLA_pmf.index <= S0 * max_moneyness]  # set a upper bound range for strike

        # divide by normalizing constant so that they sum to 1, this may not be appropriate
        BLA_pmf = BLA_pmf / BLA_pmf.sum()
        BLA_pmf = pd.DataFrame(BLA_pmf, columns=['pmf'])
        self.fitted_models['IV_Int'] = BLA_pmf
        return BLA_pmf

    def kernel_regression(self, x, y, est_pts, wb=200):
        # Gaussian Kernel
        def kernel(x):
            return np.exp(-np.power(x, 2) / 2) / np.sqrt(2 * np.pi)

        res = list()
        for et_pt in est_pts:
            distance = x - et_pt
            kernel_wgts = kernel(distance / wb)
            sum_ = np.sum(kernel_wgts)
            if sum_ == 0.0:
                res.append(np.nan)
            else:
                res.append(np.dot(kernel_wgts, y) / sum_)
        return pd.DataFrame({'x': est_pts, 'y': res}).set_index('x')

    # compute call price using BS formula
    def bs_call(self, S, K, T, r, vol):
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    def bs_put(self, S, K, T, r, vol):
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return np.exp(-r * T) * K * norm.cdf(-d2) - S * norm.cdf(-d1)

    # calculate maturity T in trading days term, require download of trading calendar from "mcal" module
    def get_T(self, maturity_str):
        # store how many days in each month
        month_last_day = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

        year = self.date[:4]
        month = self.date[4:6]
        day = self.date[6:8]
        start_date = f'{year}-{month}-{day}'

        mat = maturity_str
        mat_month = str(strptime(mat.split('-')[0], '%b').tm_mon + 100)[-2:]
        mat_year = '20' + str(int(mat.split('-')[1]) + 100)[-2:]
        mat_month_last_day = month_last_day[int(mat_month)]  # last day of the month
        end_date = f'{mat_year}-{mat_month}-{str(mat_month_last_day + 100)[-2:]}'

        hkex = mcal.get_calendar('HKEX')
        calendar_result = hkex.schedule(start_date=start_date,
                                        end_date=end_date).index  # see how many trading days left
        T = (calendar_result.shape[0] - 1) / 251
        assert T > 0, "T must be greater than 0!"
        return T

    def get_rf(self):
        year = int(self.date[:4])
        month = int(self.date[4:6])
        day = int(self.date[6:8])

        # download HIBOR rate from HKAB website
        r = requests.get(
            f'http://www.hkab.org.hk/hibor/listRates.do?lang=en&Submit=Search&year={year}&month={month}&day={day}')
        soup = BeautifulSoup(r.content, 'html5lib')
        result = soup.findAll('td', text=True)

        flag = False
        rf = 0
        for item in result:
            if flag:
                rf = float(item.text)
                break
            if item.text == '12 Months':
                flag = True
        return rf / 100


    def download_option_data(self):
        url = 'https://www.hkex.com.hk/eng/stat/dmstat/dayrpt/hsio{}.htm'.format(self.date[2:])
        html = requests.get(url).content

        if 'This page will be redirected to the homepage after five seconds' in str(html):
            expt_str = 'Option Data of date {} has been removed from HKEX'.format(self.date[2:])
            raise expt_str

        # BeautifulSoup to read html and extract option data
        soup = BeautifulSoup(html, 'html5lib')
        result = soup.findAll(text=True)
        new_l = list()
        for table in result[6:-2]:
            for i in table.split('\n'):
                if (len(i) == 0) or (i.isspace()) or ('CONTRACT STRIKE' in i) or ('MONTH' in i) or ('TOTAL' in i) or (
                        'MONTH PUT/CALL' in i) or ('CONTRACT STRIKE' in i):
                    continue
                i = i.replace('|', '')
                new_l.append(i)
        headers = '  '.join(['maturity', 'strike', 'type', 'AHOpen', 'AHHigh', 'AHLow', 'AHClose', 'AHVolume',
                             'Open', 'High', 'Low', 'Close', 'Chg', 'IV%', 'Volume',
                             'CombHigh', 'CombLow', 'CombVolume', 'OI', 'ChgOI'])
        new_l = [headers] + new_l
        CSV_IO = StringIO('\n'.join(new_l))
        df = pd.read_csv(CSV_IO, delimiter=r"\s+")
        df.strike = df.strike.astype(float).astype(int)

        return df

    # fit mixture lognormal distribution to option prices, K is the number of mixture
    def fit_mixture_lognormal(self, K, r_constr=0.2, vol_constr=0.1, evaluation_steps=5, min_hsi=5, max_hsi=100000,
                              min_moneyness=0.3, max_moneyness=2):
        # estimation points
        est_pts = np.arange(min_hsi, max_hsi, evaluation_steps)

        # find option prices of selected maturity, these values are used to compute cost of fitting.
        maturity_str = self.option_df.maturity.unique()[
            self.maturity_id]  # get the string name of maturity, for example, MAR-22
        call_option_prices = \
            self.option_df.loc[(self.option_df.maturity == maturity_str) & (self.option_df.type == "C")]['Close']
        put_option_prices = \
            self.option_df.loc[(self.option_df.maturity == maturity_str) & (self.option_df.type == "P")]['Close']
        n = len(call_option_prices) + len(put_option_prices)  # total number of real option prices
        strikes = self.filtered_df.strike  # strikes of these real options

        call_payoff_matrix, put_payoff_matrix = self.create_payoff_matirx(est_pts, strikes)

        # x = [w1, u1, sigma1, ..., w_n, u_n, sigma_n], w is weight, u & sigma are parameters in lognormal
        # S0 is HSI spot, T is maturity in years, est_pts are estimation points
        def obj_func(x, S0, est_pts, T):
            density = None
            for i in range(K):
                start_i = i * 3
                if density is None:
                    density = x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)
                else:
                    density = density + x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)
            density = density / density.sum()
            cost = self.mean_squared_error(est_pts, n, density, call_option_prices, put_option_prices,
                                           call_payoff_matrix, put_payoff_matrix)
            return cost

        S0 = self.S0
        T = self.T

        def sum_wgt_constraint1(x):
            tmp = 0
            for i in range(K):
                start_i = i * 3
                tmp = tmp + x[start_i]
            return tmp - 1

        def sum_wgt_constraint2(x):
            tmp = 0
            for i in range(K):
                start_i = i * 3
                tmp = tmp + x[start_i]
            return 1 - tmp

        consTR = [{'type': 'ineq', 'fun': sum_wgt_constraint1},
                  {'type': 'ineq', 'fun': sum_wgt_constraint2}]

        def wgt_constraint_maker(i=0):
            def constraint(x):
                return x[i]

            return constraint

        def vol_constraint_maker2(i=0):
            def constraint(x):
                return x[i + 2] - vol_constr

            return constraint

        def mean_constraint_maker(i=0):
            def constraint(x):
                return r_constr - x[i + 1]

            return constraint

        def mean_constraint_maker2(i=0):
            def constraint(x):
                return x[i + 1] + r_constr

            return constraint

        for i in range(K):
            consTR += [{'type': 'ineq', 'fun': wgt_constraint_maker(i * 3)}]
            consTR += [{'type': 'ineq', 'fun': vol_constraint_maker2(i * 3)}]
            consTR += [{'type': 'ineq', 'fun': mean_constraint_maker(i * 3)}]
            consTR += [{'type': 'ineq', 'fun': mean_constraint_maker2(i * 3)}]

        consTR = tuple(consTR)

        x0 = [1 / K, 0, 0.1] * K
        res = minimize(obj_func, np.array(x0), method="cobyla", constraints=consTR, args=(S0, est_pts, T))

        x = res.x
        density = None
        for i in range(K):
            start_i = i * 3
            if density is None:
                density = x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)
            else:
                density = density + x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)

        density = density.loc[density.index >= S0 * min_moneyness]  # set a lower bound range for strike
        density = density.loc[density.index <= S0 * max_moneyness]  # set a upper bound range for strike
        density = density / density.sum()
        density = pd.DataFrame(density)
        density.columns = ['pmf']
        self.fitted_models[f'mixture_lognorm_{K}'] = density
        return density

    # create a payoff matrix, each row is an array of terminal payoff if terminal price is est_pts, strike is k
    def create_payoff_matirx(self, est_pts, strikes):
        call_payoff_matrix = list()
        put_payoff_matrix = list()

        for k in strikes:
            call_payoff_matrix.append(np.clip(est_pts - k, a_min=0, a_max=None))

        for k in strikes:
            put_payoff_matrix.append(np.clip(k - est_pts, a_min=0, a_max=None))

        call_payoff_matrix = np.array(call_payoff_matrix)
        put_payoff_matrix = np.array(put_payoff_matrix)
        return call_payoff_matrix, put_payoff_matrix

    # calculate the predicted price of options given a RND
    def predict_price(self, density, payoff_matrix, lower_clip=1):
        # option value is the discounted expected payoff using RND
        return np.nan_to_num(
            np.clip(
                np.array(
                    np.dot(payoff_matrix, density) * np.exp(-self.rf * self.T)).reshape(-1), a_min=lower_clip,
                a_max=None))

    # calculate difference in predicted price of currenct mixture model VS actual price
    def mean_squared_error(self, est_pts, n, density, call_option_prices, put_option_prices, call_payoff_matrix,
                           put_payoff_matrix):
        call_est_err = self.predict_price(density, call_payoff_matrix) - call_option_prices
        put_est_err = self.predict_price(density, put_payoff_matrix) - put_option_prices

        sum_ = np.power(call_est_err, 2).sum() + np.power(put_est_err, 2).sum()

        mean_dif = np.dot(est_pts, density) - self.S0 * np.exp(self.rf * self.T)
        return sum_ / n

    # lognormal distribution with parameters S0, r=rf, vol, T
    def lognormal(self, S0, r, vol, T, est_pts):
        if vol <= 0:
            return pd.DataFrame(np.zeros(est_pts.shape[0]), index=est_pts)
        alpha = np.log(S0) + (r - 1 / 2 * vol ** 2) * T
        beta = vol * np.sqrt(T)
        density = 1 / (est_pts * beta * np.sqrt(2 * np.pi)) * np.exp(
            -((np.log(est_pts) - alpha) ** 2) / (2 * beta ** 2))
        density = density / density.sum()
        return pd.DataFrame(density, index=est_pts)


def CEV_European(params, alpha, beta, t_steps=500, S_steps=500, max_moneyness=2, min_moneyness=0, ):
    # implied volatility under CEV model
    def LVF_vol(alpha, St, beta):
        ret = np.zeros(St.shape[0])
        ret[1:] = alpha * np.power(St[1:], 1 - beta)
        ret[0] = np.nan
        return ret

    # calculate matrices required in Crank-Nicolson method
    def cal_matrics(dt, r, q, vol_vector, S_grids):
        dS = np.diff(S_grids)
        vol_sqrt = np.power(vol_vector, 2)
        S_j = S_grids
        S_sqrt = np.power(S_j, 2)
        dS_sqrt = np.power(dS, 2)

        sum_dS = dS[:-1] + dS[1:]
        sqrt_term = (dt * vol_sqrt[1:-1] * S_sqrt[1:-1]) / (2 * (dS_sqrt[:-1] * dS[1:] + dS_sqrt[1:] * dS[:-1]))
        rq_term = dt * (r - q) * S_j[1:-1] / (2 * sum_dS)

        b = -(sqrt_term * sum_dS) - r / 2 * dt
        c = rq_term + (sqrt_term * dS[:-1])
        a = (sqrt_term * dS[1:]) - rq_term

        B = diags([1 - b, -a[1:], -c[:-1]], [0, -1, 1]).toarray()
        C = diags([1 + b, a[1:], c[:-1]], [0, -1, 1]).toarray()

        return a, b, c, B, C

    # extract parameters for the option to price
    S0 = params['S0']
    T = params['T']
    K = params['K']
    r = params['r']
    q = params['q']
    type1 = params['type']
    r_adj = r - q

    # time (t) points in grid
    dt = T / t_steps
    t_pts = np.arange(0, T + dt, dt)

    # create non-uniform grids, which is denser when strike is close to spot, this improves efficiency
    normal_sd = np.linspace(start=-2, stop=2, num=S_steps)
    normal_density = scipy.stats.norm.pdf(normal_sd)
    inverse_density = 1 / normal_density
    dS = inverse_density / np.sum(inverse_density) * (max_moneyness - min_moneyness)
    S_grids = np.zeros(S_steps + 1)
    S_grids[1:] = dS
    S_grids = np.cumsum(S_grids)
    S_grids = np.sort(S_grids)
    S_grids = S_grids * S0

    # empty grid
    f = np.empty((len(t_pts), len(S_grids)))
    f[:] = np.nan

    # boundary conditions
    if type1 == 'Call':
        # expiry boundary conditions
        f[-1, :] = np.clip(S_grids - K, a_min=0, a_max=None)
        # upper boundary for S_max, discounted
        f[:, -1] = np.clip(S_grids[-1] - K, a_min=0, a_max=None) * np.exp(-r * (T - t_pts))
        # lower boundary for S_0
        f[:, 0] = 0

    else:
        # expiry boundary conditions
        f[-1, :] = np.clip(K - S_grids, a_min=0, a_max=None)
        # upper boundary for S_max
        f[:, -1] = 0
        # lower boundary for S_0, discounted
        f[:, 0] = np.clip(K - S_grids[0], a_min=0, a_max=None) * np.exp(-r * (T - t_pts))

    # vector of implied volatility
    vol_vector = LVF_vol(alpha, S_grids, beta)

    # calculate matrics
    a, b, c, B, C = cal_matrics(dt, r, q, vol_vector, S_grids)
    B_inv = np.linalg.inv(B)
    # probagate values from boundaries to whole grid
    for i in range(len(t_pts) - 1)[::-1]:
        f_i = f[i + 1, :]
        f_im1 = f[i, :]

        d = np.zeros(f_i.shape[0])[1:-1]
        d[0] = a[0] * (f_i[0] + f_im1[0])
        d[-1] = c[-1] * (f_i[-1] + f_im1[-1])

        next_F = np.dot(B_inv, (np.dot(C, f_i[1:-1]) + d))
        f[i, :][1:-1] = next_F

    result = pd.DataFrame({'S0': S_grids[0:].flatten(), 'val': f[0, :].flatten()})
    return result.iloc[(result['S0'] - S0).abs().argsort()[0]][1]
