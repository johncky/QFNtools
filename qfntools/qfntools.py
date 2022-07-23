import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.decomposition import PCA
from scipy.stats import entropy
import requests
from bs4 import BeautifulSoup
from io import StringIO
from statsmodels.sandbox.tools.tools_pca import pca

import requests
from io import StringIO
from bs4 import BeautifulSoup
from scipy.optimize import minimize
from scipy.stats import norm
import pandas_market_calendars as mcal
from time import strptime


class EfficientFrontier:
    def __init__(self, risk_measure, alpha=5, entropy_bins=None):
        self.risk_measure = risk_measure.lower()
        self.df = None

        # percentile for VaR
        self.alpha = alpha
        self.entropy_bins = entropy_bins

        # covariance matrix
        self.omega = None

        # mean vector
        self.R = None

        self.wgt = None
        self.mu_range = None
        self.risk_range = None

    def fit(self, df, wbnd, mu_range):
        self.R = df.mean()
        self.omega = df.cov()
        self.df = df

        if self.risk_measure == 'cvar':
            obj_func = self.cvar
        elif self.risk_measure == 'var':
            obj_func = self.var
        elif self.risk_measure == 'entropy':
            obj_func = self.entropy_
        else:
            obj_func = self.sd

        risk_range = np.zeros(len(mu_range))
        n = df.shape[1]
        wgt = list()

        for i in range(len(mu_range)):
            mu = mu_range[i]

            # initial weight = equal weight
            x_0 = np.ones(n) / n

            # bounds for weightings
            bndsa = [wbnd for j in range(n)]

            # constraint 1 --> type=equality --> sum(weightings) = 1
            # constraint 2 --> type=equality --> np.dot(w^T, R) = mu
            consTR = (
                {'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}, {'type': 'eq', 'fun': lambda x: mu - np.dot(x, self.R)})

            # Find min risk portfolio for given mu
            w = minimize(obj_func, x_0, method='SLSQP', constraints=consTR, bounds=bndsa)

            risk_range[i] = obj_func(w.x)

            wgt.append(np.squeeze(w.x))

        wgt = np.array(wgt)
        self.wgt = wgt
        self.mu_range = mu_range
        self.risk_range = risk_range

    def sd(self, w):
        return np.dot(w, np.dot(self.omega, w.T))

    def cvar(self, w):
        ret = np.dot(self.df, w.T)
        return abs(min((np.mean(ret[ret <= np.percentile(ret, self.alpha)]), 0)))

    def var(self, w):
        ret = np.dot(self.df, w.T)
        return abs(min(np.percentile(ret, self.alpha), 0))

    def entropy_(self, w):
        ret = np.dot(self.df, w.T)
        ret = pd.Series(ret)
        if self.entropy_bins is None:
            bins = int(np.sqrt(ret.shape[0] / 5))
        else:
            bins = self.entropy_bins
        counts = ret.value_counts(bins=bins)
        h = entropy(counts)
        return h

    def weights(self, drop_zero_col=True, rounding=True):
        df = pd.DataFrame(self.wgt, index=[self.mu_range, self.risk_range], columns=self.df.columns)
        df.index.names = ['mu', self.risk_measure]
        if rounding:
            df = np.round(df, 2)
        if drop_zero_col:
            df = df.loc[:, (df != 0.0).any(axis=0)]
        return df

    def to_csv(self, path):
        self.weights().to_csv(path)

    def plot(self, port_only=True):
        risk_range = self.risk_range
        mu_range = self.mu_range
        if self.risk_measure == 'cvar':
            label = '{}% Conditional VaR (%)'.format(self.alpha)
        elif self.risk_measure == 'var':
            label = '{}% VaR (%)'.format(self.alpha)
        elif self.risk_measure == 'entropy':
            label = 'Entropy'
        else:
            risk_range = np.sqrt(risk_range)
            label = 'Standard Deviation (%)'
        if port_only:
            wgt = self.weights()
            idx = (wgt != 0).sum(1) != 1
            risk_range = risk_range[idx]
            mu_range = mu_range[idx]
        plt.plot(np.multiply(risk_range, 100), np.multiply(mu_range, 100), color="red")
        plt.xlabel(label, fontsize=10)
        plt.ylabel("Expected Return (%)", fontsize=10)
        plt.title("Efficient Frontier", fontsize=12)


class DynamicBeta:
    def __init__(self):
        self.kf = None
        self.filter_df = None
        self.smoothed_df = None
        self.x = None
        self.y = None

    def fit(self, y, x, factor_pca=False, n_pc=3):
        if type(y) == pd.Series:
            y = pd.DataFrame(y)
        if type(x) == pd.Series:
            x = pd.DataFrame(x)

        if factor_pca:
            xreduced, factors, evals, evecs = pca(x, keepdim=n_pc)
            x = pd.DataFrame(factors, index=x.index, columns=['PC{}'.format(i) for i in range(1, n_pc + 1)])

        n_dim_obs = y.shape[1]
        n_dim_state = x.shape[1] + 1
        ntimestep = y.shape[0]
        factors = sm.add_constant(x)

        fac_obs = np.array(factors)
        obs_matrics = np.zeros((ntimestep, n_dim_obs, n_dim_state))

        for i in range(n_dim_obs):
            obs_matrics[:, i, :] = fac_obs

        kf = KalmanFilter(n_dim_obs=n_dim_obs, n_dim_state=n_dim_state, transition_matrices=np.eye(factors.shape[1]),
                          observation_matrices=obs_matrics,
                          em_vars=['transition_covariance', 'observation_covariance', 'initial_state_mean',
                                   'initial_state_covariance'])

        if factor_pca:
            cols = ['Intercept'] + ['beta-PC{}'.format(i) for i in range(1, n_pc + 1)]
        else:
            cols = ['Intercept'] + list(x.columns)

        self.kf = kf
        filter_state_means, filter_state_covs = kf.filter(y)

        self.filter_df = pd.DataFrame(filter_state_means, index=y.index, columns=cols)
        smoothed_state_means, smoothed_state_covs = kf.smooth(y)
        self.smoothed_df = pd.DataFrame(smoothed_state_means, index=y.index, columns=cols)

    def plot(self, smoothed=False):
        if smoothed:
            self.smoothed_df.plot()
        else:
            self.filter_df.plot()


class EigenPortfolio:
    def __init__(self, n_pc):
        self.n_pc = n_pc
        self.rets = None
        self.norm_wgt = None
        self.explained_variance_ratio = None

    def fit(self, rets):
        self.rets = rets
        std = rets.std(0)
        std_rets = (rets - rets.mean(0)) / std
        pca = PCA(n_components=self.n_pc, random_state=1)
        pca.fit(std_rets)
        norm_wgt = pd.DataFrame(pca.components_, columns=rets.columns,
                                index=['PC{}'.format(i + 1) for i in range(self.n_pc)]).T
        norm_wgt = norm_wgt.div(std, axis=0)
        self.norm_wgt = norm_wgt / norm_wgt.sum()
        self.explained_variance_ratio = pca.explained_variance_ratio_

    def price(self):
        port_df = self.return_().add(1).cumprod()
        return port_df

    def plot(self):
        port_df = self.price()
        port_df.plot()

    def return_(self):
        ret_df = pd.DataFrame(np.dot(self.rets, self.norm_wgt), index=self.rets.index,
                              columns=['PC{}'.format(i + 1) for i in range(self.n_pc)])
        return ret_df


class FactorSelection:
    def __init__(self, req_exp, req_corr, max_f_cor):
        self.req_exp = req_exp
        self.req_corr = req_corr
        self.max_f_cor = max_f_cor
        self.eigen_port = None
        self.fac = list()
        self.x = None
        self.R2 = None
        self.betas = None

    def fit(self, y, x):
        self.x = x
        self.eigen_port = EigenPortfolio(self.req_exp)
        self.eigen_port.fit(y)
        fac_id = list()
        fac_p = list()
        eigen_port_df = self.eigen_port.return_()

        for p in range(eigen_port_df.shape[1]):
            epi = eigen_port_df.iloc[:, p]
            for f in range(x.shape[1]):
                if f in fac_id:
                    continue
                r, p = pearsonr(x.iloc[:, f], epi)
                if abs(r) >= self.req_corr:
                    fac_id.append(f)
                    fac_p.append(abs(p))

        sort_fac = [x for _, x in sorted(zip(fac_p, fac_id))]

        if len(fac_id) == 0:
            print('All factors < req_corr')
            return

        removed = list()
        for i in range(len(sort_fac) - 1):
            if sort_fac[i] in removed:
                continue
            for j in range(i + 1, len(sort_fac)):
                if sort_fac[j] in removed:
                    continue
                r, p = pearsonr(x.iloc[:, sort_fac[i]], x.iloc[:, sort_fac[j]])

                if abs(r) > self.max_f_cor:
                    removed.append(sort_fac[j])
        sort_fac = [x for x in sort_fac if x not in removed]
        self.fac = list(x.columns[sort_fac])
        self.build_model()

    def merged_df(self):
        eigen_port_df = self.eigen_port.return_(True)
        merged_df = pd.concat([self.factor_df(), eigen_port_df], axis=1)
        return merged_df

    def factor_df(self):
        return self.x.copy()[self.fac]

    def plot_eigen(self, const_rebal=False):
        self.eigen_port.plot(const_rebal)

    def build_model(self):
        fac_df = self.factor_df()
        fac_df = (fac_df - fac_df.mean()) / fac_df.std()
        eqty_df = self.eigen_port.df.copy()
        eqty_df = (eqty_df - eqty_df.mean()) / eqty_df.std()

        X = fac_df.to_numpy()
        X = sm.add_constant(X)
        R2 = list()
        betas = list()
        # Run regression on Equity returns using selected factors as predictors
        for i in range(eqty_df.shape[1]):
            model = sm.OLS(endog=eqty_df.iloc[:, i], exog=X)
            result = model.fit()
            R2.append(result.rsquared)
            betas.append(result.params)

        R2 = pd.DataFrame(R2, index=eqty_df.columns, columns=['R squared'])
        betas = pd.DataFrame(betas, index=eqty_df.columns)
        betas = betas.T
        betas.index = ['intercept'] + list(fac_df.columns)
        self.R2 = R2
        self.betas = betas.T


class HsiRND:
    def __init__(self, date, maturity_id, option_type='C'):
        self.date = date
        self.maturity_id = maturity_id
        assert (option_type in ('C', 'P')), 'Option Type must be "C" or "P"'
        self.option_type = option_type

        # download raw option data from HKEX
        self.option_df = self.download_option_data()

        # filter option data to only selected ones
        maturity_str = self.option_df.maturity.unique()[
            maturity_id]  # get the string name of maturity, for example, MAR-22
        self.filtered_df = \
            self.option_df.loc[(self.option_df.maturity == maturity_str) & (self.option_df.type == self.option_type)][
                ['strike', 'IV%', 'Close']]

        # calculate spot HSI S0, maturity T in years (trading days only), risk-free rate rf
        self.T = self.get_T(maturity_str=maturity_str)
        self.rf = self.get_rf()  # 1-year HIBOR is taken as risk-free
        if self.option_type == 'C':
            self.S0 = (self.filtered_df['Close'] + self.filtered_df.strike).iloc[0]
        else:
            self.S0 = (self.filtered_df['Close'] + self.filtered_df.strike).iloc[-1]

        # store fitted pdf/pmf
        self.fitted_models = dict()

    def fit_BLA(self, evaluation_steps=5, min_hsi=5, max_hsi=100000):
        filtered_df = self.filtered_df

        # keep track of changes in strike gap
        k = filtered_df['strike']
        gap_chg = k.diff().diff().fillna(0)

        # identify the index of strike where change in strike gap occurs
        idx_gap_chg = gap_chg.loc[gap_chg != 0.0].index

        # calculate probability mass function from option prices, pmf is also the butterfly spread prices
        price = filtered_df['Close']
        # formula for butterfly spread prices
        bfly_spr = (-2 * price + price.shift(1) + price.shift(-1))

        # Prices of Butterfly spread with these strikes cannot be calculated, as prices of some strikes are not available (changing strike gap)
        for i in idx_gap_chg:
            bfly_spr.loc[i - 1] = 0

        # divide by strike difference to obtain pmf
        pmf = bfly_spr / k.diff().fillna(1)
        pmf.index = k

        # enforce monotonicity in cmf, remove negative pmf using interpolation
        fixed_pmf = self.enforce_cmf_monotonicity(pmf)
        fixed_pmf = pd.DataFrame(fixed_pmf, columns=['pmf'])

        # sum to 1
        fixed_pmf = fixed_pmf / fixed_pmf.sum()
        self.fitted_models['BLA'] = fixed_pmf
        return fixed_pmf

    # given a density in pd.Series with index=strikes, values=pmf, it remove negative pmf and interpolate cmf using monotonic interpolation
    def enforce_cmf_monotonicity(self, density):
        strikes = density.index
        cmf = density.cumsum()
        pos_cmf = cmf.loc[cmf.diff() >= 0]
        pos_cmf = pos_cmf.loc[pos_cmf.diff() >= 0]

        # interpolate & calculate pmf
        fixed_pmf = pos_cmf.reindex(strikes).interpolate(method='pchip').diff().fillna(0)
        return fixed_pmf

    def fit_IV(self, wb_mult=4, evaluation_steps=5, min_hsi=5, max_hsi=100000, min_moneyness=0.3, max_moneyness=2):
        filtered_df = self.filtered_df

        # estimation points
        est_pts = np.arange(min_hsi, max_hsi, evaluation_steps)

        # run kernel regression on Black-Scholes Implied Volatility
        kernel_iv = self.kernel_regression(x=filtered_df['strike'], y=filtered_df['IV%'].values, est_pts=est_pts,
                                           wb=200 * wb_mult)
        kernel_iv = kernel_iv.fillna(method='ffill').fillna(method='bfill').clip(
            lower=1)  # BS IV cannot be 0, set lower bound to 1

        # convert Black-Scholes IV to option value through BSF, and calculate RND using BLA
        S0 = self.S0
        if self.option_type == 'C':
            prices = pd.Series(self.bs_call(S0, kernel_iv.index.values, self.T, self.rf, (kernel_iv['y'] / 100).values),
                               index=kernel_iv.index)
            BLA_pmf = (-2 * prices + prices.shift(1) + prices.shift(-1)).fillna(0) / evaluation_steps
        else:
            prices = pd.Series(self.bs_put(S0, kernel_iv.index.values, self.T, self.rf, (kernel_iv['y'] / 100).values),
                               index=kernel_iv.index).diff().diff()
            BLA_pmf = (-2 * prices + prices.shift(1) + prices.shift(-1)).fillna(0) / evaluation_steps

        BLA_pmf = self.enforce_cmf_monotonicity(BLA_pmf)
        BLA_pmf = BLA_pmf.loc[BLA_pmf.index >= S0 * min_moneyness]  # set a lower bound range for strike
        BLA_pmf = BLA_pmf.loc[BLA_pmf.index <= S0 * max_moneyness]  # set a upper bound range for strike

        BLA_pmf = BLA_pmf / BLA_pmf.sum()
        BLA_pmf = pd.DataFrame(BLA_pmf, columns=['pmf'])
        self.fitted_models['IV_Int'] = BLA_pmf
        return BLA_pmf

    def kernel_regression(self, x, y, est_pts, wb=200):
        # Gaussian Kernel
        def kernel(x):
            return np.exp(-np.power(x, 2) / 2) / np.sqrt(2 * np.pi)

        res = list()
        for et_pt in est_pts:
            distance = x - et_pt
            kernel_wgts = kernel(distance / wb)
            sum_ = np.sum(kernel_wgts)
            if sum_ == 0.0:
                res.append(np.nan)
            else:
                res.append(np.dot(kernel_wgts, y) / sum_)
        return pd.DataFrame({'x': est_pts, 'y': res}).set_index('x')

    def bs_call(self, S, K, T, r, vol):
        N = norm.cdf
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)

    def bs_put(self, S, K, T, r, vol):
        N = norm.cdf
        d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        return np.exp(-r * T) * K * norm.cdf(-d2) - S * norm.cdf(-d1)

    # calculate maturity T in trading days term, require download of trading calendar from "mcal" module
    def get_T(self, maturity_str):
        # store how many days in each month
        month_last_day = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

        year = self.date[:4]
        month = self.date[4:6]
        day = self.date[6:8]
        start_date = f'{year}-{month}-{day}'

        mat = maturity_str
        mat_month = str(strptime(mat.split('-')[0], '%b').tm_mon + 100)[-2:]
        mat_year = '20' + str(int(mat.split('-')[1]) + 100)[-2:]
        mat_month_last_day = month_last_day[int(mat_month)]  # last day of the month
        end_date = f'{mat_year}-{mat_month}-{str(mat_month_last_day + 100)[-2:]}'

        hkex = mcal.get_calendar('HKEX')
        calendar_result = hkex.schedule(start_date=start_date,
                                        end_date=end_date).index  # see how many trading days left
        T = (calendar_result.shape[0] - 1) / 251
        assert T > 0, "T must be greater than 0!"
        return T

    def get_rf(self):
        year = int(self.date[:4])
        month = int(self.date[4:6])
        day = int(self.date[6:8])

        # download HIBOR rate from HKAB website
        r = requests.get(
            f'http://www.hkab.org.hk/hibor/listRates.do?lang=en&Submit=Search&year={year}&month={month}&day={day}')
        soup = BeautifulSoup(r.content, 'html5lib')
        result = soup.findAll('td', text=True)

        flag = False
        for item in result:
            if flag:
                rf = float(item.text)
                break
            if item.text == '12 Months':
                flag = True
        return rf / 100

    def select_maturity(self, ascending_order, evaluation_steps=5, min_hsi=5, max_hsi=100000):
        self.create_payoff_matirx()
        self.forward_price = self.S0 * np.exp(self.rf * self.T)

    def download_option_data(self):
        url = 'https://www.hkex.com.hk/eng/stat/dmstat/dayrpt/hsio{}.htm'.format(self.date[2:])
        html = requests.get(url).content

        if 'This page will be redirected to the homepage after five seconds' in str(html):
            expt_str = 'Option Data of date {} has been removed from HKEX'.format(self.date[2:])
            raise expt_str

        # BeautifulSoup to read html and extract option data
        soup = BeautifulSoup(html, 'html5lib')
        result = soup.findAll(text=True)
        new_l = list()
        for table in result[6:-2]:
            for i in table.split('\n'):
                if (len(i) == 0) or (i.isspace()) or ('CONTRACT STRIKE' in i) or ('MONTH' in i) or ('TOTAL' in i) or (
                        'MONTH PUT/CALL' in i) or ('CONTRACT STRIKE' in i):
                    continue
                i = i.replace('|', '')
                new_l.append(i)
        headers = '  '.join(['maturity', 'strike', 'type', 'AHOpen', 'AHHigh', 'AHLow', 'AHClose', 'AHVolume',
                             'Open', 'High', 'Low', 'Close', 'Chg', 'IV%', 'Volume',
                             'CombHigh', 'CombLow', 'CombVolume', 'OI', 'ChgOI'])
        new_l = [headers] + new_l
        CSV_IO = StringIO('\n'.join(new_l))
        df = pd.read_csv(CSV_IO, delimiter=r"\s+")
        df.strike = df.strike.astype(float).astype(int)

        return df

    # fit mixture lognormal distribution to option prices, K is the number of mixture
    def fit_mixture_lognormal(self, K, r_constr=0.2, vol_constr=0.1, evaluation_steps=5, min_hsi=5, max_hsi=100000,
                              min_moneyness=0.3, max_moneyness=2):
        # estimation points
        est_pts = np.arange(min_hsi, max_hsi, evaluation_steps)

        # find option prices of selected maturity, these values are used to compute cost of fitting.
        maturity_str = self.option_df.maturity.unique()[
            self.maturity_id]  # get the string name of maturity, for example, MAR-22
        call_option_prices = \
            self.option_df.loc[(self.option_df.maturity == maturity_str) & (self.option_df.type == "C")]['Close']
        put_option_prices = \
            self.option_df.loc[(self.option_df.maturity == maturity_str) & (self.option_df.type == "P")]['Close']
        n = len(call_option_prices) + len(put_option_prices)  # total number of real option prices
        strikes = self.filtered_df.strike  # strikes of these real options

        call_payoff_matrix, put_payoff_matrix = self.create_payoff_matirx(est_pts, strikes)

        # x = [w1, u1, sigma1, ..., w_n, u_n, sigma_n], w is weight, u & sigma are parameters in lognormal
        # S0 is HSI spot, T is maturity in years, est_pts are estimation points
        def obj_func(x, S0, est_pts, T):
            density = None
            for i in range(K):
                start_i = i * 3
                if density is None:
                    density = x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)
                else:
                    density = density + x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)
            density = density / density.sum()
            cost = self.mean_squared_error(est_pts, n, density, call_option_prices, put_option_prices,
                                           call_payoff_matrix, put_payoff_matrix)
            return cost

        S0 = self.S0
        T = self.T

        def sum_wgt_constraint1(x):
            tmp = 0
            for i in range(K):
                start_i = i * 3
                tmp = tmp + x[start_i]
            return tmp - 1

        def sum_wgt_constraint2(x):
            tmp = 0
            for i in range(K):
                start_i = i * 3
                tmp = tmp + x[start_i]
            return 1 - tmp

        consTR = [{'type': 'ineq', 'fun': sum_wgt_constraint1},
                  {'type': 'ineq', 'fun': sum_wgt_constraint2}]

        def wgt_constraint_maker(i=0):
            def constraint(x):
                return x[i]

            return constraint

        def vol_constraint_maker2(i=0):
            def constraint(x):
                return x[i + 2] - vol_constr

            return constraint

        def mean_constraint_maker(i=0):
            def constraint(x):
                return r_constr - x[i + 1]

            return constraint

        def mean_constraint_maker2(i=0):
            def constraint(x):
                return x[i + 1] + r_constr

            return constraint

        for i in range(K):
            consTR += [{'type': 'ineq', 'fun': wgt_constraint_maker(i * 3)}]
            consTR += [{'type': 'ineq', 'fun': vol_constraint_maker2(i * 3)}]
            consTR += [{'type': 'ineq', 'fun': mean_constraint_maker(i * 3)}]
            consTR += [{'type': 'ineq', 'fun': mean_constraint_maker2(i * 3)}]

        consTR = tuple(consTR)

        x0 = [1 / K, 0, 0.1] * K
        res = minimize(obj_func, np.array(x0), method="cobyla", constraints=consTR, args=(S0, est_pts, T))

        x = res.x
        density = None
        for i in range(K):
            start_i = i * 3
            if density is None:
                density = x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)
            else:
                density = density + x[start_i] * self.lognormal(S0, x[start_i + 1], x[start_i + 2], T, est_pts)

        density = density.loc[density.index >= S0 * min_moneyness]  # set a lower bound range for strike
        density = density.loc[density.index <= S0 * max_moneyness]  # set a upper bound range for strike
        density = density / density.sum()
        density = pd.DataFrame(density)
        density.columns = ['pmf']
        self.fitted_models[f'mixture_lognorm_{K}'] = density
        return density

    # create a payoff matrix, each row is an array of terminal payoff if terminal price is est_pts, strike is k
    def create_payoff_matirx(self, est_pts, strikes):
        call_payoff_matrix = list()
        put_payoff_matrix = list()

        for k in strikes:
            call_payoff_matrix.append(np.clip(est_pts - k, a_min=0, a_max=None))

        for k in strikes:
            put_payoff_matrix.append(np.clip(k - est_pts, a_min=0, a_max=None))

        call_payoff_matrix = np.array(call_payoff_matrix)
        put_payoff_matrix = np.array(put_payoff_matrix)
        return call_payoff_matrix, put_payoff_matrix

    # calculate the predicted price of options given a RND
    def predict_price(self, density, payoff_matrix, lower_clip=1):
        # option value is the discounted expected payoff using RND
        return np.nan_to_num(
            np.clip(
                np.array(
                    np.dot(payoff_matrix, density) * np.exp(-self.rf * self.T)).reshape(-1), a_min=lower_clip,
                a_max=None))

    # calculate difference in predicted price of currenct mixture model VS actual price
    def mean_squared_error(self, est_pts, n, density, call_option_prices, put_option_prices, call_payoff_matrix,
                           put_payoff_matrix):
        call_est_err = self.predict_price(density, call_payoff_matrix) - call_option_prices
        put_est_err = self.predict_price(density, put_payoff_matrix) - put_option_prices

        sum_ = np.power(call_est_err, 2).sum() + np.power(put_est_err, 2).sum()

        mean_dif = np.dot(est_pts, density) - self.S0 * np.exp(self.rf * self.T)
        return sum_ / n

    # lognormal distribution with parameters S0, r=rf, vol, T
    def lognormal(self, S0, r, vol, T, est_pts):
        if vol <= 0:
            return pd.DataFrame(np.zeros(est_pts.shape[0]), index=est_pts)
        alpha = np.log(S0) + (r - 1 / 2 * vol ** 2) * T
        beta = vol * np.sqrt(T)
        density = 1 / (est_pts * beta * np.sqrt(2 * np.pi)) * np.exp(
            -((np.log(est_pts) - alpha) ** 2) / (2 * beta ** 2))
        density = density / density.sum()
        return pd.DataFrame(density, index=est_pts)
