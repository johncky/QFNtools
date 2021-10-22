# Quant Finance

### 1. Long-only Efficient Frontier
There is analytical solution for finding Efficient Frontier of a collection of assets. However, solution portfolios are usually not tradable. 
Because we cannot short certain assets, or take extremely high leverage through long-short.
In this section, "scipy.optimize.minimize" function is used to find the long-only EF for a self-defined collection of assets. These portfolios are tradable.

Key:
- Utilize scipy.optimize.minimize to find the long-only mean-variance optimal portfolios from a group of stocks
- Bounds for weightings to be [lb, ub] determine the type of EF we want. [0,1] bound gives long-only EF. [-inf, inf] gives EF that allows short-selling.
- Long-only EF are not smooth

Result:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/Efficient_Frontier(NoRiskFree).png?raw=true)

1. A long-only Efficient Frontier is found
2. weightings for efficient portfolios are output in csv.

### 2. Factor Analysis on Global Equities Indices using PCA
In this section, I want to reconcile factors of global equity indices with real assets that represent flow/bullishness/risk-aversion.
For example, if there a factor that represent the contrast of developed vs emerging markets, what commodities/currency/pairs can represent this effect?

Key:
- Perform PCA on indices return to find principal components that explains most variations in indices returns. Then construct eigen portfolios from those PCs.
- Interpret the first 3 PCs: What do they represent? (global base return, developed market effect, locational difference...)
- Run linear regression on each PC with each factor(commodities (oil, copper, coin,...), currency, currency pair) to find a set of factors that represent the PC of indices return.
- Run linear regression on indices return with the selected factors from the last step, and interpret.

Result:

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/2_1.png?raw=true)

1. The first PC seems to be a general level of global returns, all have the same signs with mean of 21% 
2. The second PC seems to reflect a contrast of developed markets vs emerging markets
3. The third PC seems to reflect the locational difference between fast-growing Asian economies vs the western world
4. We seem not able to assign themes to the remaining PCs

5. Below are the eigen portfolios of global indices (top 3)

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/2_3.png?raw=true)

6. AUD/JPY is the most important factor for explaining SPX return variation. 
   This is very intuitive. Since the pair represent global bullishness (AUD rise=commodity flow, JPY down = flow to equity)

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/2_2.png?raw=true)

### 3. Eigen Portfolio 
Reconstruct eigen portfolios of liquid US ETF. 
There is Hindsight bias, I calculated the PC loadings using all data, and reconstruct their price path using the PC loadings
The chart is NOT the actual portfolio value of investing in the eigen portfolios.

Result:

1. Eigen Portfolios 

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_EP.png?raw=true)

2. Eigen Portfolios have uncorrelated returns

![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/3_uncor.png?raw=true)

