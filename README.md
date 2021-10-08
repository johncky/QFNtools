# Quant Finance

### 1. Long-only Efficient Frontier
- Utilize scipy.optimize.minimize to find the long-only mean-variance optimal portfolios from a group of stocks
- Set bounds for weightings to be [-1000, 1000] will give efficient frontier that allows short-selling. Although the EF will be smooth, some portfolios are not tradable.
