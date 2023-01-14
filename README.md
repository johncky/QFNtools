Content
=============================


**Some functions**:
- [Efficient Frontier](#efficient-frontier) : Solve Efficient Frontier with risk measures "Variance", "Entropy", "Conditional VaR" or "VaR". 

- [Dynamic Beta](#dynamic-beta) ：Find betas/hedge ratio in dynamic factor model with Kalman Filter.

- [Risk Neutral Density](#hsi-risk-neutral-density) ：Download option data from HKEX and find underlying risk-neutral densities.

- [Bayesian Linear Regression with Bayesian Model Averaging](#Bayesian-linear-regression-with-Bayesian-model-averaging ) ： Run Bayesian linear regression
with Bayesian model averaging, sample from posterior distri

- [Pricing European options under CEV model with non-uniform Crank-Nicolson Method](#bayesian-model-averaging) ： Run Bayesian linear regression
with Bayesian model averaging, sample from posterior distri
- 
**Other stuff**:
- [Market neutral mean reversion arbitrage](#market-neutral-arbitrage) 
- [Option Pricing Under CEV Model using FDM with Non-uniform Discretization](#option-pricing-under-cev-model-using-fdm-with-non-uniform-discretization)


### Bayesian Linear Regression
Run Bayesian linear regression with different priors
```python
from qfntools.qfntools import BayesLinReg, ZellnerLinReg, UnitInfoLinReg

data = pd.read_table('http://www2.stat.duke.edu/~pdh10/FCBS/Exercises/azdiabetes.dat', sep="\s+").drop(
   columns=['diabetes'])
data.insert(0, 'intercept', 1)

# user-defined prior for beta
BLR = BayesLinReg()
BLR.fit(data, formula="skin ~ intercept + bmi + age", beta0=np.zeros(3), lambda0=np.identity(3), sigma02=0.2, v0=1)

# Zellner's g prior
BLR = ZellnerLinReg(g=100)
BLR.fit(data, formula="skin ~ intercept + bmi + age", sigma02=0.2, v0=1)

# Unit information prior
BLR = UnitInfoLinReg()
BLR.fit(data, formula="skin ~ intercept + bmi + age", sigma02=0.2, v0=1)

BLR.fit(data, formula="skin ~ intercept + bmi + age", sigma02=0.2, v0=1)
```

#### MCMC samples
```python
# return a diagnostic object
mcmc_samples = BLR.sample_posterior(10000)
# retreieve actual samples DataFrame
mcmc_samples.samples
```
#### MCMC diagnostic plots
```python
# autocorrelation plot to check chain stickiness & mixing
mcmc_samples.acf()

# traceplot to check convergence & mixing
mcmc_samples.traceplot()
```

#### Posterior distribution
```python
# retreieve actual samples DataFrame
mcmc_samples.samples

# plot posterior density
mcmc_samples.density_plot()

# posterior mean
mcmc_samples.mean()

# posterior credible interval of parameters
mcmc_samples.credible_interval()

# approximated effective sample size 
mcmc_samples.ess()
```
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/posterior_density.png?raw=true)

####  Predictive distribution
```python
# predictive posterior check
mcmc_samples.ppc_plot( X, y, max_sample=100) # max_sample: maximum number of simulated datasets to plot

# predict conditional mean: E[Y_new|X_new, X_data]
mcmc_samples.predict(X_new)

# credible interval of Y_new | X_new
mcmc_samples.predictive_ci(X_new, q=(0.025,0.975))
```
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/ppc.png?raw=true)

## Bayesian linear regression with Bayesian model averaging 
Run Bayesian linear regression (Zellner's g prior) with Bayesian model averaging. 

```python
from qfntools.qfntools import BMA_LinReg
data = pd.read_table('http://www2.stat.duke.edu/~pdh10/FCBS/Exercises/azdiabetes.dat', sep="\s+").drop(columns=['diabetes'])
data.insert(0,'intercept', 1)



# Bayesian linear regression (Zellner's g prior) with Bayesian model averaging 
BMA = BMA_LinReg(g=10000)
BMA.fit(data, y='skin', v0=1, sigma02=0.2)
```

#### MCMC samples
```python
samples = BMA.sample_posterior(10000) # model & parameter (betas, sigma2) samples from Gibbs sampler
```

#### Diagnostic plots
```python
BMA.diagnostic()
```
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/BMA_diagnostic?raw=true)



## Efficient Frontier
Solve Efficient Frontier of assets. Risk measures can be "standard deviation", "Entropy",  "Conditional VaR", "VaR"

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cmin_%7Bw%7D%20%5Cquad%20%26%0ARiskMeasure(w%2C%20X)%5C%5C%0A%5Ctextrm%0A%7Bs.t.%7D%20%5Cquad%20%26%0A%5Cmu%5E%7BT%7D%20%20w%20%3D%20%5Cmu_%7Btarget%7D%5C%5C%20%5Cquad%20%26%0A%5Csum_%7B1%7D%5E%7Bn%7Dw_i%20%3D1%5C%5C%0A%26ub%20%5Cgeq%20w%5Cgeq%20lb%20%20%20%20%5C%5C%0A%5Cend%7Balign*%7D">

```python
from qfntools.qfntools import EfficientFrontier

# risk_measure: one of "sd", "entropy", "var", "cvar"
# alpha: percentile for "cvar" and "var" 
ef = EfficientFrontier(risk_measure='cvar', alpha=5)

# wbnd: bound of weightings; (0,1): long-only, (None, None): allow short-sale
# mu_range: range of target return to optimize. Tune this.
ef.fit(asset_return_df, wbnd=(0, 1), mu_range=np.arange(0.0055, 0.013, 0.0002))
```

#### Diagnostics :
```python
# plot efficient frontier
ef.plot() 

# retreive weights
ef.weights()
```

## Dynamic Beta
Use Kalman Filter to estimate betas in dynamic factor model. Cast a dynamic factor model into linear state space form (factor loadings as state variable, return y as observable variable), and use Kalman filter and smoother to estimate the loadings.

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cbeta_%7Bt%2B1%7D%20%3D%20I%20%5Cbeta_%7Bt%7D%20%2B%20%5Cepsilon_%7Bt%2B1%7D%5C%5C%0A%5Cy_%7Bt%7D%20%3D%20x_%7Bt%7D%5ET%20%5Cbeta_%7Bt%7D%20%2B%20%5Cvarepsilon_%7Bt%2B1%7D%5C%5C%0Ax_%7B0%7D%20~%20N(%5Cmu_0%2C%20%5CSigma_0)%5C%5C%0A%5Cepsilon_%7Bt%7D%20~%20N(0%2C%20Q)%5C%5C%0A%5Cvarepsilon_%7Bt%7D%20~%20N(0%2C%20R)%5C%5C%0A%5Cend%7Balign*%7D%0A">

In filtering, expectation and covariance at time t is estimated given new observations at time t.

In smoothing, expectation and covariance at previous times are estimated given observations up to time T.

```python
from qfntools.qfntools import DynamicBeta

dfe = DynamicBeta()

# y: returns of one/more assets (Y) in pd.Dataframe
# x: factors values (X) in pd.Dataframe
# factor_pca: bool, default=False. If True, principal components of X are used as factors.
# n_pc: int, number of principal components of X to be used as factors. 
dfe.fit(yRet_df, factors_df)
```

#### Diagnostics
```python
# plot KF smoother values
dfe.plot(smoothed=True)

# plot KF filter values
dfe.plot(smoothed=False)
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/4_filterbetas.png?raw=true)

## Option Pricing under CEV model using FDM with non-uniform discretization
A stock follows constant elasticity of variance (CEV) model if the following is satisfied:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/CEV_model.jpg?raw=true)

CEV model allows a more realistic non-constant Black-Scholes Implied Volatility (IV) curve across option strikes.
In practice, we often notice that IV for options are higher at extreme ends, forming a "smile" shape curve.

When alpha=stock sigma, Beta=1, CEV becomes the Black-Scholes Model (BSM), and option price converges
to Black-Scholes Formula price.

#### Usage
```python
from qfntools.qfntools import CEV_European

# parameters of the option
# T: maturity in years
# K: strike of the option
# r: risk-free rate
# q: dividend yield
# S0: spot price
# type: 'Call' or 'Put'

option_1 = {'T': 1, 'K': 0.8*100, 'r':0.02, 'q':0.01, 'S0': 100, 'type':'Call'}

# price the option under CEV(alpha=20, beta=2) model
CEV_European(option_1, alpha=20, beta=2)
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/Option_value_under_CEV.png?raw=true)


## HSI Risk Neutral Density
Web-scrape HSI option prices from HKEX websites, and use various methods to extract risk-neutral density (RND) from option prices

```python
from qfntools.qfntools import HsiRND

# date: date of option prices, in the format "yyyymmdd"
# maturity_id: int, represents maturity of options. 0 indicates the closest maturity options, 1 indicates the second closest maturity etc
# option_type: "C" or "P" (call or put options). Indicates the options to use for extracting RND
rnd = HsiRND(date='20220721', maturity_id=3, option_type='C')

# fit BLA method 
rnd.fit_BLA(plot=True)

# use kernel regression to interpolate implied volatility curve, then use BS formula to inversely compute option prices of a continuum of 
# strikes, and finally use BLA method to extract RND
# wb_mult: bandwidth multiplier for kernel regression of BS implied volatility. bandwidth = wb_mult * 200 hsi points.
rnd.fit_IV(wb_mult=2)

# model terminal HSI as a mixture of log-normal density, and find the mixture that 'best fit' current option prices
# K: number of log-normal distribution in the mixture model. default 2.
rnd.fit_mixture_lognormal(K=2)
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/rnd_shift.jpg?raw=true)
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/diff_mat_rnd.jpg?raw=true)


## Market Neutral Arbitrage
Paper: ["Statistical Arbitrage in the U.S. Equities Market"](https://github.com/johncky/Quantitative-Finance/blob/main/paper/Statistical_Arbitrage_in_the_U.S._Equities_Market.pdf)

Implementation: ["Market neutral Arbitrage"](https://github.com/johncky/Quantitative-Finance/blob/main/strategies/Mean_reversion_arb.ipynb)

[Stock data](https://github.com/johncky/Quantitative-Finance/blob/main/data/mean_reversion_data.zip)
, [Performance Report](https://github.com/johncky/Quantitative-Finance/blob/main/strategies/Mean_reversion_result.html)


General idea: Decompose stock returns into systematic components, idiosyncratic drift and residuals, model systematic factors with 15 principal components:

r = alpha + B1 * PC1 + ... + B15 * PC15  + dX

X (residuals) is modelled as mean-reversion process:

dX = k*(m - X) dt + sigma dW

When residuals X > m, dX has (-ve) expectation. When X < m, dX has (+ve) expectation

Strategies:
1) PCA on SP500 + Nasdaq100 stocks, find the top 15 principal components
2) Run linear regression on each stock with the PCs to determine their hedge ratios:
   r = alpha + B1 * r_pc1 + ... + B15 * r_pc15  + dX
3) estimate parameters in X(t): k, m, sigma_equilibrium
4) calculate the standardized s-score: s = (X - m) / sigma_eq
5) if s > 1.25, short stock, hedge with $B1 PC1, $B2
   of PC2, ... etc.  (each PC is a portfolio of stocks).
6) Close short when s<0.75.
7) Similarly, long when s<-1.25, close long when s>-0.75.

Maintain a 4x gross leverage.


![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/arb_beta.png?raw=true)
