# Quant Finance

### 1. Long-only Efficient Frontier
#### There is analytical solution for finding Efficient Frontier of a collection of assets. However, that solution is usually not tradable. Since we may not be able to short certain assets, or to take extremely high leverage through long-short. So in this section, we use optimize method to find the long-only EF for a self-defined collection of assets.
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
1. The first PC seems to be a general level of global returns, all have the same signs with mean of 21%# The second PC seems to reflect a contrast of developed markets vs emerging markets
2. The third PC seems to reflect the locational difference between fast-growing Asian economies vs the western world
3. We seem not able to assign themes to the remaining PCs
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/PCs.png?raw=true)
4. AUD/JPY is the most important factor for explaining SPX return variation. This is very intuitive. Since the pair represent global buillishness (AUD rise =commodity flow, JPY down = flow to equity)
![alt text](https://github.com/johncky/Quantitative-Finance/blob/main/pic/PC2.png?raw=true)
