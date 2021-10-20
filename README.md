# Quant Finance

### 1. Long-only Efficient Frontier
#### There is analytical solution for finding Efficient Frontier of a collection of assets. However, that solution is usually not tradable. Since it is sometimes to short certain assets, or to take extremely high leverage through long-short. So in this section, we use optimize method to find the long-only EF for a self-defined collection of assets.
Key:
- Utilize scipy.optimize.minimize to find the long-only mean-variance optimal portfolios from a group of stocks
- Set bounds for weightings to be [-1000, 1000] will give efficient frontier that allows short-selling. Although the EF will be smooth, some portfolios are not tradable.

### 2. Factor model for Global Equities Indices through PCA
#### Note: This may not be the factor model you are looking for. In this section, we want to reconcile the fundamental driving force of different global equity indices with some real assets that represent somethings. For example, we want to know if there are effects from developed vs emerging markets? What commodities/currency/pairs can represent this effect?
Key:
- Perform PCA on equities return to find principal components that explains most variations in indices returns, then construct eigen portfolios
- Interpret the first 3 PC: What do they represent?
- Run linear regression on each PC for each factor in a collection (we use oil, commodities, currency index, currency pair that represent bullishness / flow). This is to find out a set of real asset factors that represent the PCA of equities most closely.
- Run linear regression on indices return for the factors we selected in last step, and interpret.
Result:

