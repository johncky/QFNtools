Content
=============================

**Reports**:
- [Market neutral mean reversion arbitrage](#market-neutral-arbitrage) 
- [HSI Risk Neutral Density](#hsi-risk-neutral-density)
- [Option Pricing Under CEV Model using Non-uniform Discretization](#option-pricing-under-cev-model-using-non-uniform-discretization-fdm)
- [Option Pricing using Monte Carlo Simulation & Binomial Trees]()

**Tools**:
- [Efficient Frontier](#efficient-frontier) : Solve Efficient Frontier with risk measures "Variance", "Entropy", "Conditional VaR" or "VaR". 

- [Dynamic Beta](#dynamic-beta) ：Find betas/hedge ratio in dynamic factor model with Kalman Filter.

- [Factor Selection](#factor-selection) ：Select "factors" and build factor model to explain returns.

- [Eigen Portfolio](#eigen-portfolio) ：Find eigen portfolios of assets.

- [Risk Neutral Density](#hsi-risk-neutral-density) ：Download option data from HKEX and find underlying risk-neutral densities.


## Efficient Frontier
Solve Efficient Frontier of assets. Risk measures can be "standard deviation", "Entropy",  "Conditional VaR", "VaR"

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cmin_%7Bw%7D%20%5Cquad%20%26%0ARiskMeasure(w%2C%20X)%5C%5C%0A%5Ctextrm%0A%7Bs.t.%7D%20%5Cquad%20%26%0A%5Cmu%5E%7BT%7D%20%20w%20%3D%20%5Cmu_%7Btarget%7D%5C%5C%20%5Cquad%20%26%0A%5Csum_%7B1%7D%5E%7Bn%7Dw_i%20%3D1%5C%5C%0A%26ub%20%5Cgeq%20w%5Cgeq%20lb%20%20%20%20%5C%5C%0A%5Cend%7Balign*%7D">


```python
from qfntools.qfntools import EfficientFrontier

ef = EfficientFrontier(risk_measure='cvar', alpha=5)
ef.fit(asset_return_df, wbnd=(0, 1), mu_range=np.arange(0.0055, 0.013, 0.0002))
```

#### \_\_init\_\_(_risk\_measure_, _alpha_) :
risk_measure:
one of "sd", "entropy", "var", "cvar"

alpha:
percentile for "cvar" and "var" 

entropy_bins:
no of bins for entropy

#### fit(_df_, _wbnd_, _mu\_range_) :

df:
returns of assets in pd.DataFrame

wbnd:
bound of weightings; (0,1): long-only, (None, None): allow short-sale

mu_range:
range of target return to optimize. Tune this.

#### plot() :
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/1_EF(cvar).png?raw=true)


#### weights() :
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/1_weights.png?raw=true)

## Dynamic Beta
Use Kalman Filter to estimate betas in dynamic factor model

Cast a dynamic factor model into linear state space form (factor loadings as state variable, return y as observable variable), and use Kalman filter and smoother to estimate the loadings.

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cbeta_%7Bt%2B1%7D%20%3D%20I%20%5Cbeta_%7Bt%7D%20%2B%20%5Cepsilon_%7Bt%2B1%7D%5C%5C%0A%5Cy_%7Bt%7D%20%3D%20x_%7Bt%7D%5ET%20%5Cbeta_%7Bt%7D%20%2B%20%5Cvarepsilon_%7Bt%2B1%7D%5C%5C%0Ax_%7B0%7D%20~%20N(%5Cmu_0%2C%20%5CSigma_0)%5C%5C%0A%5Cepsilon_%7Bt%7D%20~%20N(0%2C%20Q)%5C%5C%0A%5Cvarepsilon_%7Bt%7D%20~%20N(0%2C%20R)%5C%5C%0A%5Cend%7Balign*%7D%0A">

In filtering, expectation and covariance at time t is estimated given new observations at time t.

In smoothing, expectation and covariance at previous times are estimated given observations up to time T.

```python
from qfntools.qfntools import DynamicBeta

dfe = DynamicBeta()
dfe.fit(yRet_df, factors_df)
```

#### fit(_y_, _x_, _factor_pca_, _n\_pc_):
y:
returns of one/more assets (Y) in pd.Dataframe. 

x:
factors values (X) in pd.Dataframe. 

factor_pca:
bool, default=False. If True, principal components of X are used as factors.

n_pc:
int, number of principal components of X to be used as factors. 

#### plot(_smoothed_):
smoothed:
bool, default=False. If True, plot result from smoother. If False, plot result from filter.

filter:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/4_filterbetas.png?raw=true)

smoother:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/4_dynamicbetas.png?raw=true)


## Factor Selection

Select "factors" from assets (X), ues them as independent variables in factor model.

      1. Find Principal Components of Y

      2. Select factors from X whose absolute correlation with the PCs >= "req_corr". Use them
      to represent PCs of Y. 
      
      3. If between-factor correlation >= "max_f_cor", remove the one with lower correlation with PCs.
      
      4. Build factor model: Y ~ Intercept + (B1 * X1) + (B2 * X2) + ...

```python
from qfntools.qfntools import FactorSelection

fs = FactorSelection(req_exp=0.8, req_corr=0.4, max_f_cor=0.7)
fs.fit(y=equity_return_df, x=factor_df)
```

#### \_\_init\_\_(_req\_exp_, _req\_corr_, _max\_f\_cor_) :
req_exp:
[0,1], required explanatory power of the PCs.

req_corr:
[0,1], required absolute correlation of factor with PCs.

max_f_cor:
[0,1], maximum between-factor correlation.

#### fit(_y_, _x_) :
y:
returns of assets in pd.DataFrame

x:
returns of potential factors in pd.DataFrame

#### Results:
   ```python
fs.betas
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_model.png?raw=true)

```python
fs.R2
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_r2.png?raw=true)

## Eigen Portfolio
Find eigen portfolios of assets X. 

```python
from qfntools.qfntools import EigenPortfolio

ep = EigenPortfolio(req_exp=0.8)
ep.fit(asset_return_df)
```

#### \_\_init\_\_(_req\_exp_):
req_exp:
[0,1], required explanatory power. Determines the number of Eigen Portfolios


#### fit(_df_):
df:
asset returns in pd.DataFrame

#### Results :
   ```python
    ep.plot()
    ep.price()
    ep.return_()
```

## Market Neutral Arbitrage
Backtested with backtrader, 2013-01-01 to 2021-06-01, using {Nasdaq 100 + SP500} stocks, 10bps
commission. Result: 0 beta, sharpe 1.04

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


![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/arb_ev.png?raw=true)

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/arb_beta.png?raw=true)

## HSI Risk Neutral Density
A variety of methods have been devised to extract risk-neutral density (RND) from European option prices.
These extracted RND can be used to track changes in expected moments of terminal price and
gauge changes in market sentiment.

Details of each method are in 
[Project Report](https://github.com/johncky/Quantitative-Finance/blob/main/paper/rnd_project.pdf)



```python
from qfntools.qfntools import HsiRND

rnd = HsiRND(date='20220721', maturity_id=3)
rnd.fit_BLA()
rnd.fit_IV()
rnd.fit_mixture_lognormal(K=2)
```

#### \_\_init\_\_(_date_, _maturity_id_, _option_type_):
date:
date of option prices, in the format "yyyymmdd"

maturity_id:
integer from 0 to around 11, represents maturity of options. 0 indicates the closest maturity options, 1 indicates 
the second closest maturity etc

option_type:
"C" or "P" (call or put options). Indicates the options to use for the Breeden and Litzenberger Approach

#### fit_BLA(_plot_):
plot:
bool, if true plot RND.
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/BLA_rnd_plot.png?raw=true)


#### fit_IV(wb_mult, _plot_):
wb_mult:
bandwidth multiplier for kernel regression of BS implied volatility. bandwidth = wb_mult * 200 hsi points.

plot:
bool, if true plot RND.
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/IV_int_rnd_plot.png?raw=true)

#### fit_mixture_lognormal(K, _plot_):
K:
number of lognormal distribution in the mixture model. default 2.

plot:
bool, if true plot RND.
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/mixture_rnd_plot.png?raw=true)

### Results:
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/rnd_shift.jpg?raw=true)
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/diff_mat_rnd.jpg?raw=true)


## Option Pricing under CEV model using non-uniform discretization FDM
A stock follows constant elasticity of variance (CEV) model if the following is satisfied:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/CEV_model.jpg?raw=true)

CEV model allows a more realistic non-constant Black-Scholes Implied Volatility (IV) curve across option strikes.
In practice, we often notice that IV for options are higher at extreme ends, forming a "smile" shape curve.

(Note: when alpha=sigma, Beta=1, CEV becomes the famous Black-Scholes Model)

In this [Report](https://github.com/johncky/Quantitative-Finance/blob/main/paper/option_pricing_project.pdf)
, we assumes stock dynamics to follow CEV, and price European options using Crank–Nicolson method with a non-uniform
discretization. 


### Results:
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/Option_value_under_CEV.png?raw=true)

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/CNM_convergence_1.png?raw=true)

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/CNM_convergence_2.png?raw=true)

When alpha=stock sigma, Beta=1, CEV becomes the Black-Scholes Model (BSM), and option price converges 
to Black-Scholes Formula price:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/CNM_CEV_convergence_to_BSF.png?raw=true)
