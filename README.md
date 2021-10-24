Content
=============================

- [Efficient Frontier](#efficient-frontier)
- [Factor Selection](#factor-selection)
- [Eigen Portfolio](#eigen-portfolio)

[Jupyter Notebook](https://github.com/johncky/Quantitative-Finance/blob/main/explanatory_notebook): explanatory notebooks

# Efficient Frontier
Solve Efficient Frontier of a group of assets. Risk measures can be set to "standard deviation", "Conditional VaR", "VaR"

### Example:

```python
    from qfntools import EfficientFrontier
   
    ef = EfficientFrontier(risk_measure='cvar', alpha=5)
    ef.fit(asset_return_df, wbnd=(0,1), mu_range=np.arange(0.0055,0.013,0.0002))
```

## Class & Functions:
### EfficientFrontier(_risk\_measure_, _alpha_) :
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

# Factor Selection
Select "factors" from a group of factor assets (X), ues them as predictors to build Factor Models to explain returns of another group (Y).

Selection Process:

1. Find Principal Components of a group of assets (Y)
2. Select factors from another group of factor assets (X) whose absolute correlation with the PCs >= "req_corr". Use them
   to represent the fictional PCs. (i.e. use real assets to represent the fictional Eigen portfolios)

3. If between-factor correlation >= "max_f_cor", remove the one that has lower correlation with PCs.
4. Run linear regression on each asset from Y with the selected factors from X. 
   
Hopefully, asset returns from Y can be explained by factor assets returns X.

Example:

```python
    from qfntools import FactorSelection
    
    fs = FactorSelection(req_exp=0.8, req_corr=0.4, max_f_cor=0.7)
    fs.fit(y=equity_return_df, x=factor_return_df)
```

## Class & Functions:
### FactorSelection(_req\_exp_, _req\_corr_, _max\_f\_cor_) :
**req_exp**:
[0,1], required explanatory power of the PCs. larger value = more PCs.

**req_corr**:
[0,1], required absolute correlation of factor with PCs in order to be selected.

**max_f_cor**:
[0,1], maximum allowed between-factor correlation.

### fit(_y_, _x_) :
**y**:
df, group of assets to be explained

**x**:
df, group of "factor" assets X used to explain returns of Y

### factor model :
   ```python
    fs.betas
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_model.png?raw=true)

### model R squared :
```python
    fs.R2
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_r2.png?raw=true)

### selected factors :
   ```python
    fs.factor_df()
```

# Eigen Portfolio
Find eigen portfolios of a group of assets. Compute their returns and price paths.

Example:

```python
    from qfntools import EigenPortfolio

    ep = EigenPortfolio(req_exp=0.8)
    ep.fit(asset_return_df)
```

## Class & Functions:
### EigenPortfolio(_req\_exp_):
**req_exp**:
required explanatory power, between 0 and 1. This determines the number of Principal Components / Eigen Portfolios


### fit(_df_):
**df**:
df, asset returns


### price(_const\_rebal_) / return\_ (_const\_rebal_):
**const_rebal**:
bool. If False, invest weights of eigen portfolios at period start. However, return correlation
of eigen portfolios will not be exactly zero. If True, weights are maintained every period, returns of eigen portfolios have zero correlations.

```python
    ef.price()
    ef.return_()
```

### plot :
   ```python
    ep.plot()
```

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/2_3.png?raw=true)


