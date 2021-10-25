Content
=============================

- [Efficient Frontier](#efficient-frontier) : Solve **Efficient Frontier** with risk measures "Variance", "Conditional VaR" or "VaR". 

- [Dynamic Beta](#dynamic-beta) ：Find **dynamic betas** in a factor model with **Kalman Filter**.

- [Factor Selection](#factor-selection) ：Select "factors" and build **factor model** to explain returns.

- [Eigen Portfolio](#eigen-portfolio) ：Find **eigen portfolios** of a group of assets.

Usage Example:
- [Dynamic beta in CAPM](https://github.com/johncky/Quantitative-Finance/blob/main/example/Dynamic_beta_in_CAPM.ipynb):
  Find dynamic beta of an asset in CAPM, in which factor is market return.
  
- [Dynamic beta(s) in factor model](https://github.com/johncky/Quantitative-Finance/blob/main/example/Dynamic_beta_in_factor_model.ipynb):
  Find dynamic beta(s) of an/a group of asset(s), in which factors are currencies/commodities/bonds etc.

[Jupyter Notebook](https://github.com/johncky/Quantitative-Finance/blob/main/explanatory_notebook): explanatory notebooks


# Efficient Frontier
Solve Efficient Frontier of a group of assets. Risk measures can be set to "standard deviation", "Conditional VaR", "VaR"

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cmin_%7Bw%7D%20%5Cquad%20%26%0ARiskMeasure(w%2C%20X)%5C%5C%0A%5Ctextrm%0A%7Bs.t.%7D%20%5Cquad%20%26%0A%5Cmu%5E%7BT%7D%20%20w%20%3D%20%5Cmu_%7Btarget%7D%5C%5C%20%5Cquad%20%26%0A%5Csum_%7B1%7D%5E%7Bn%7Dw_i%20%3D1%5C%5C%0A%26ub%20%5Cgeq%20w%5Cgeq%20lb%20%20%20%20%5C%5C%0A%5Cend%7Balign*%7D">

### Example:

```python
from qfntools import EfficientFrontier

ef = EfficientFrontier(risk_measure='cvar', alpha=5)
ef.fit(asset_return_df, wbnd=(0,1), mu_range=np.arange(0.0055,0.013,0.0002))
```

## Methods:
### \_\_init\_\_(_risk\_measure_, _alpha_) :
**risk_measure**:
one of "sd", "var", "cvar". choose the risk measure to minimize for target mean return. (note: var is NOT coherent risk measure)

**alpha**:
percentile for "cvar" and "var" calculations

### fit(_df_, _wbnd_, _mu\_range_) :

**df**:
df, returns of assets

**wbnd**:
bound of weightings; (0,1) means long-only, (None, None) means short-selling allowed

**mu_range**:
Tune this! range of target return to optimize. If this is above / below possible target return achieved, corresponding risk measures are not sensible.

### plot :
   ```python
ef.plot()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/1_EF(cvar).png?raw=true)


### weights :
```python
ef.weights()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/1_weights.png?raw=true)

# Dynamic Beta
Use **Kalman Filter** to estimate **dynamic betas**  in a factor model.

i.e. it models factor exposure beta as state variable and return y as observable variable in a linear state-space model.

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cbeta_%7Bt%2B1%7D%20%3D%20I%20%5Cbeta_%7Bt%7D%20%2B%20%5Cepsilon_%7Bt%2B1%7D%5C%5C%0A%5Cy_%7Bt%7D%20%3D%20x_%7Bt%7D%5ET%20%5Cbeta_%7Bt%7D%20%2B%20%5Cvarepsilon_%7Bt%2B1%7D%5C%5C%0Ax_%7B0%7D%20~%20N(%5Cmu_0%2C%20%5CSigma_0)%5C%5C%0A%5Cepsilon_%7Bt%7D%20~%20N(0%2C%20Q)%5C%5C%0A%5Cvarepsilon_%7Bt%7D%20~%20N(0%2C%20R)%5C%5C%0A%5Cend%7Balign*%7D%0A">

In Kalman filtering, it finds expectation and covariance of y at time t given new observation of observable variable
at time t.

In Kalman smoothing, it finds expectation and covariance of y at previous time given observations up to time T.

Example:

```python
from qfntools import DynamicBeta

dfe = DynamicBeta()
dfe.fit(yRet_df, factors_df)
```

## Methods:
### fit(_y_, _x_, _factor_pca_, _n\_pc_):
**y**:
df, returns of one/more assets (Y), dependent variable in the factor model. If there are more one asset, the dependent variable is a random vector.

**x**:
df, factors X, independent variables in the factor model.

**factor_pca**:
bool (Default=False). If True, principal components of X are used as factors. If False, X is used as factors.

**n_pc**:
int, number of principal components of X to be used as factors. Only useful when factor_pca = True.

### plot(_smoothed_):
**smoothed**:
bool (Default=False). If True, plot result from Kalman Smoother. If False, plot result from Kalman Filter.

```python
dfe.plot(smoothed=False)
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/4_filterbetas.png?raw=true)

```python
dfe.plot(smoothed=True)
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/4_dynamicbetas.png?raw=true)


# Factor Selection

Select "factors" from a group of factor assets (X), ues them as independent variables in factor model.

      Selection Process:

      1. Find Principal Components of Y

      2. Select factors from X whose absolute correlation with the PCs >= "req_corr". Use them
      to represent PCs of Y. (e.g. use real assets return to represent PCs of Y)
      
      3. If between-factor correlation >= "max_f_cor", remove the one that has lower correlation with PCs.
      
      4. Build factor model: Y ~ Intercept + (B1 * X1) + (B2 * X2) + ...

### Example:

```python
from qfntools import FactorSelection

fs = FactorSelection(req_exp=0.8, req_corr=0.4, max_f_cor=0.7)
fs.fit(y=equity_return_df, x=factor_df)
```

## Methods:
### \_\_init\_\_(_req\_exp_, _req\_corr_, _max\_f\_cor_) :
**req_exp**:
[0,1], required explanatory power of the PCs. larger value = more PCs.

**req_corr**:
[0,1], required absolute correlation of factor with PCs in order to be selected.

**max_f_cor**:
[0,1], maximum allowed between-factor correlation.

### fit(_y_, _x_) :
**y**:
df, returns of assets to be explained

**x**:
df, group of "factors"  to explain y

### Results:
   ```python
fs.betas
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_model.png?raw=true)

```python
fs.R2
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_r2.png?raw=true)

### see selected factors :
   ```python
fs.factor_df()
```

# Eigen Portfolio
Find eigen portfolios of a group of assets X. 

### Example:

```python
from qfntools import EigenPortfolio

ep = EigenPortfolio(req_exp=0.8)
ep.fit(asset_return_df)
```

## Methods:
### \_\_init\_\_(_req\_exp_):
**req_exp**:
required explanatory power, between 0 and 1. This determines the number of Principal Components / Eigen Portfolios


### fit(_df_):
**df**:
df, asset returns


### price(_const\_rebal_) / return\_ (_const\_rebal_):
**const_rebal**:
bool. If False, invest weights of eigen portfolios at period start. However, correlation
of eigen portfolios will not be exactly zero. If True, weights are maintained and eigen portfolios have zero correlations.

### plot :
   ```python
    ep.plot()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/2_3.png?raw=true)

### eigen portfolios:
```python
    ef.price()
    ef.return_()
```