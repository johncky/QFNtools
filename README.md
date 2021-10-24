# QFNtools

## Content
1. Efficient Frontier: Solve Efficient Frontier of a group of assets. Risk measures can be set to "standard deviation", "Conditional VaR", "VaR"
2. Factor Selection: Select "factors" from a group of factor assets (X), ues them to build Factor Models to explain returns of a group of assets (Y).
3. Eigen Portfolio: Find eigen portfolios of a group of assets. Compute their returns and price paths.

### 1. Efficient Frontier

Example:

```python
    from qfntools import EfficientFrontier
   
    # Preferably weekly or monthly
    data = pd.read_excel('./data/biggestETFData.xlsx',index_col = 0)
    data = data.resample('M').last()
    data_return= data.pct_change().dropna(how = 'all')
   
    ef = EfficientFrontier(risk_measure='cvar', alpha=5)
    ef.fit(data_return, wbnd=(0,1), mu_range=np.arange(0.0055,0.013,0.0002))
```

plot:
   ```python
    ef.plot()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/1_EF(cvar).png?raw=true)


weights:
```python
    ef.weights()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/1_weights.png?raw=true)

### Class & Functions:
#### _EfficientFrontier(risk_measure, alpha)_:
risk_measure: one of "sd", "var", "cvar". choose the risk measure to minimize for target mean return. (note: var is NOT coherent risk measure)

alpha: percentile to use in "cvar" and "var" calculation

#### _EfficientFrontier.fit(df, wbnd, mu_range)_:

df: df, returns of assets

wbnd: bound of weightings; (0,1) means long-only, (None, None) means short-selling allowed

mu_range (Tune this): range of target return to optimize. If this is above / below possible target return achieved, corresponding risk measures are not sensible. 

### 2. Factor Selection
Selection Process:

1. Find Principal Components of a group of assets (Y)
2. Select factors from another group of factor assets (X) whose absolute correlation with the PCs >= "req_corr". Use them
   to represent the fictional PCs. (i.e. use real assets to represent the fictional Eigen portfolios)

3. If between-factor correlation >= "max_f_cor", remove the one that has lower correlation with PCs.
4. Run linear regression on each asset from Y with the selected factors from X. 
   
Hopefully, asset returns from Y can be explained by factor assets returns X.

Example:

```python
# read data
    from qfntools import FactorSelection
    
    equity = pd.read_excel('./data/data2.xlsx',index_col=0, sheet_name='equity')
    factor = pd.read_excel('./data/data2.xlsx',index_col=0, sheet_name='factor')
    
    # 24 "factor" assets: commodities (oil, corn, ...), currency, currency pair, bond, ... 
    factor_return = factor.pct_change().dropna()
    
    # group of equity
    equity_return = equity.pct_change().dropna()
    
    fs = FactorSelection(req_exp=0.8, req_corr=0.4, max_f_cor=0.7)
    fs.fit(y=equity_return, x=factor_return)
```

selected factors:
   ```python
    fs.factor_df()
```

merged df of selected factors & Pprincipal components:
   ```python
    # see how each "factor" correlated with the PC of equities
    fs.merged_df().corr()
```

Factor model using selected factor
   ```python
    fs.betas
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_model.png?raw=true)


R2 of Factor Model:
```python
    fs.R2
```

It actually explains index returns pretty well ! (except China A share):

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_r2.png?raw=true)

### Class & Functions:
#### _FactorSelection(req_exp, req_corr, max_f_cor)_:
req_exp: [0,1], required explanatory power of the PCs. larger value = more PCs.

req_corr:[0,1], required absolute correlation of factor with PCs in order to be selected.

max_f_cor:[0,1], maximum allowed between-factor correlation.

#### _fit(y, x)_:
y: df, group of assets to be explained

x: df, group of "factor" assets X used to explain returns of Y


### 3. Eigen Portfolio

Example:

```python
    from qfntools import EigenPortfolio
   
    data = pd.read_excel('./data/data2.xlsx',index_col = 0)
    data_return = data.pct_change().dropna(how = 'all')
   
    ep = EigenPortfolio(req_exp=0.8)
    ep.fit(data_return)
```

plot:
   ```python
    ep.plot()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/2_3.png?raw=true)


price/return:
```python
    ef.price()
    ef.return_()
```

### Class & Functions:
#### _EigenPortfolio(req_exp)_:
req_exp: required explanatory power, between 0 and 1. This determines the number of Principal Components / Eigen Portfolios


#### _fit(df)_:
df: df, asset returns

#### _price(const_rebal) / return_(const_rebal)_:
const_rebal: bool. If False, invest weights of eigen portfolios at period start. However, return correlation
of eigen portfolios will not be exactly zero. If True, weights are maintained every period, returns of eigen portfolios have zero correlations.

For testing eigen investing strategy (investing in eigen portfolios), use const_rebal=False for checking real return.



