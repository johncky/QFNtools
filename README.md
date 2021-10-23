# QFNtools

Analytical tools for quantitative finance people.

## Content
1. Efficient Frontier: Solve efficient frontier of a selection of assets. Risk measures can be set to "standard deviation", "Conditional VaR", "VaR"
2. Eigen Portfolio: Find eigen portfolios of a selection of assets. Compute their return and price path.
3. Factor Selection: Select "factors" from a group of factor assets (X), and build Factor Models to explain returns of a group of assets (Y).

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

####EfficientFrontier(risk_measure, alpha):
risk_measure: one of "sd", "var", "cvar". choose the risk measure to minimize for target mean return. (note: var is NOT coherent risk measure)

alpha: percentile to use in "cvar" and "var" calculation

####fit(df, wbnd, mu_range):
df: pandas DataFrame of asset returns

wbnd: bound of weightings; (0,1) means long-only, (None, None) means short-selling allowed

mu_range (Tune this): range of target return to optimize. If this is above / below possible target return achieved, corresponding risk measures are not sensible. 

### 2. Factor Selection
What it does:

1. Find Principal Components of the group of assets
2. Select factors from another group of factor assets whose absolute correlation with the PCs >= "req_corr". Use them
   to represent the fictional PCs.

3. If between-factor correlation >= "max_f_cor", remove the one that has lower correlation with PCs.
4. Run linear regression on each asset with the selected factors. Factor returns can explain return of asset.

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

merged df of selected factors & Principal Components the Y (equity):
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

It actually explains indices return pretty well !:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_r2.png?raw=true)


####FactorSelection(req_exp, req_corr, max_f_cor):
req_exp: [0,1], required explanatory power of the PCs. larger value = more PCs.

req_corr:[0,1], required absolute correlation of factor with PCs in order to be selected.

max_f_cor:[0,1], maximum allowed between-factor correlation.

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

####EigenPortfolio(req_exp):
req_exp: required explanatory power, between 0 and 1. This determines the number of Principal Components / Eigen Portfolios


####fit(df):
df: pandas DataFrame of asset returns

####price(const_rebal) / return_(const_rebal):
const_rebal: bool. If False, invest weights of eigen portfolios at period start. However, return correlation
of eigen portfolios will not be exactly zero. If True, weights are maintained every period, returns of eigen portfolios have zero correlations.

For testing eigen investing strategy (investing in eigen portfolios), use const_rebal=False for checking real return.



