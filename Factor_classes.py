import json

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

class StockData:
    def __init__(self, start_date, end_date):
        self.index_ticker = "FTSEMIB.MI"
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = self.load_tickers()
        self.data = self.download_data(self.tickers)
        self.volume_data = self.download_volume_data(self.tickers)
        self.index_data = self.download_index_data(self.index_ticker)
        self.monthly_data = self.resample_data(self.data)
        self.monthly_volume_data = self.resample_data(self.volume_data)
        self.index_monthly_data = self.resample_data(self.index_data)
        self.returns = self.calculate_returns(self.monthly_data)
        self.index_returns = self.calculate_returns(self.index_monthly_data)
        
    def load_tickers(self):
        with open('Tickers.json', 'r') as json_file:
            tickers_json = json.load(json_file)
        return tickers_json['tickers']
    
    def download_data(self, tickers):
        return yf.download(tickers, start=self.start_date, end=self.end_date)['Open']
    
    def download_volume_data(self, tickers):
        return yf.download(tickers, start=self.start_date, end=self.end_date)['Volume']
    
    def download_index_data(self, index_ticker):
        return yf.download(index_ticker, start=self.start_date, end=self.end_date)['Open']
    
    def resample_data(self, data):
        return data.resample('ME').first()
    
    def calculate_returns(self, data):
        return data.pct_change().dropna()
    

class CorrelationMatrix:
    def __init__(self, returns):
        self.returns = returns
        self.correlation_matrix = self.calculate_correlation_matrix()
        
    def calculate_correlation_matrix(self):
        return self.returns.corr()
    
    def plot_heatmap(self):
        plt.figure(figsize=(20, 16))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Heatmap of Returns')
        plt.show()


class ComponentsAnalysis:
    def __init__(self, returns):
        self.returns = returns
        self.R = returns.values
        self.t, self.n = returns.shape
        self.Q_tilde = self.calculate_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.perform_eigendecomposition()
        self.explained_variance_ratio = self.calculate_explained_variance_ratio()
        self.cumulative_explained_variance = self.calculate_cumulative_explained_variance()
    
    def calculate_covariance_matrix(self):
        return (1 / self.t) * self.R @ self.R.T #asset returns covariance matrix (in t x t not n x n, nello screenshot assume R che ha t come colonne)
    
    def perform_eigendecomposition(self):
        return np.linalg.eigh(self.Q_tilde)
    
    def calculate_explained_variance_ratio(self):
        return self.eigenvalues[::-1] / np.sum(self.eigenvalues)
    
    def calculate_cumulative_explained_variance(self):
        return np.cumsum(self.explained_variance_ratio)
    
    def plot_scree(self):
        plt.figure(figsize=(20, 6))
        plt.plot(np.arange(1, len(self.eigenvalues) + 1), self.eigenvalues[::-1], 'o-', markersize=8)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title('Scree Plot')
        plt.grid(True)
        plt.show()
    
    def plot_cumulative_explained_variance(self):
        plt.figure(figsize=(20, 6))
        plt.plot(np.arange(1, len(self.eigenvalues) + 1), self.cumulative_explained_variance, 'o-', markersize=8)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.grid(True)
        plt.show()


class APCA:
    def __init__(self, returns, convergence_threshold=1e-3, max_iterations=1000):
        self.returns = returns
        self.R = returns.values
        self.t, self.n = returns.shape
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.Q_tilde = self.calculate_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.perform_eigendecomposition()
        self.explained_variance_ratio = self.calculate_explained_variance_ratio()
        self.cumulative_explained_variance = self.calculate_cumulative_explained_variance()
        self.m = self.number_factors(0.95)
        self.U_m_final, self.F_final, self.B_final = self.iterative_estimation()
    
    def calculate_covariance_matrix(self):
        return (1 / self.t) * self.R @ self.R.T
    
    def perform_eigendecomposition(self):
        return np.linalg.eigh(self.Q_tilde)
    
    def number_factors(self, threshold):
        return np.searchsorted(self.cumulative_explained_variance, threshold) + 1
    
    def calculate_explained_variance_ratio(self):
        return self.eigenvalues[::-1] / np.sum(self.eigenvalues)
    
    def calculate_cumulative_explained_variance(self):
        return np.cumsum(self.explained_variance_ratio)
    
    def iterative_estimation(self):
        previous_Delta_squared = np.inf

        for iteration in range(self.max_iterations):
            U_m = self.eigenvectors[:, -self.m:]
            F = U_m.T # Factor returns
            B = self.R.T @ U_m # Factor exposures
            Gamma = self.R.T - B @ F # Specific returns
            Delta_squared = (1 / self.t) * np.diag(Gamma @ Gamma.T) # Specific covariance matrix

            if np.all(np.abs(Delta_squared - previous_Delta_squared) < self.convergence_threshold):
                #print(f"Converged after {iteration + 1} iterations")
                break

            previous_Delta_squared = Delta_squared
            Delta_inv = np.diag(1 / np.sqrt(Delta_squared))
            R_star = Delta_inv @ self.R.T
            Q_tilde_star = (1 / self.n) * R_star.T @ R_star
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(Q_tilde_star)

        else:
            print("Did not converge within the maximum number of iterations")

        U_m_final = self.eigenvectors[:, -self.m:]
        F_final = U_m_final.T
        B_final = self.R.T @ U_m_final

        return U_m_final, F_final, B_final
    

class PortfolioWeights:
    def __init__(self, factor_returns):
        self.factor_returns = factor_returns
    
    def risk_parity_weights(self):
        cov_matrix = np.cov(self.factor_returns.T)
        inv_vols = 1 / np.sqrt(np.diag(cov_matrix))
        initial_weights = inv_vols / np.sum(inv_vols)
        
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib
            return np.sum((risk_contrib - portfolio_vol / len(weights)) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        
        return result.x
    
    def momentum_based_weights(self, lookback_period=12):
        momentum = np.mean(self.factor_returns[-lookback_period:], axis=0)
        positive_momentum = momentum.clip(min=0)
        weights = positive_momentum / np.sum(positive_momentum)
        
        return weights
    
    def tail_risk_parity_weights(self, alpha=0.05):
        cov_matrix = np.cov(self.factor_returns.T)
        
        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            tail_risks = norm.ppf(alpha) * np.sqrt(np.diag(cov_matrix))
            risk_contrib = weights * tail_risks
            return np.sum((risk_contrib - portfolio_vol / len(weights)) ** 2)
        
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        
        return result.x
    
    def random_forest_weights(self, random_seed=42):
        np.random.seed(random_seed)
        X = self.factor_returns[:-1]
        y = self.factor_returns[1:]
        tscv = TimeSeriesSplit(n_splits=5)
        val_mse_list = []
        train_mse_list = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_seed)
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            train_mse = mean_squared_error(y_train, train_predictions)
            val_mse = mean_squared_error(y_val, val_predictions)
            train_mse_list.append(train_mse)
            val_mse_list.append(val_mse)

        model.fit(X, y)
        weights = model.predict(np.mean(X, axis=0).reshape(1, -1))[0]
        weights = weights / np.sum(weights)
        return weights
    

class RollingAPCAStrategy:
    def __init__(self, data_returns, window_size, max_iterations, initial_capital=10000, transaction_cost=0.001, slippage=0.001):
        self.data_returns = data_returns
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.weight_methods = ['equal', 'inverse_volatility', 'risk_parity', 'momentum', 'tail_risk_parity', 'random_forest']
        self.portfolio_capital_dict = {}
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def rolling_apca_strategy(self, weight_method):
        train_factor_returns = []
        train_factor_loadings = []
        test_index = []
        portfolio_capital = []
        capital = self.initial_capital

        for start in range(len(self.data_returns) - self.window_size):
            long_return = 0
            short_return = 0

            end = start + self.window_size
            train_returns = self.data_returns.iloc[start:end]  # t x n
            test_returns = self.data_returns.iloc[end:end + 1]
            factor_model = APCA(train_returns, max_iterations=self.max_iterations)
            factor_returns = factor_model.F_final  # m x t
            factor_volatility = np.std(factor_returns, axis=0)
            factor_loadings = factor_model.B_final  # n x m

            portfolio_weights = PortfolioWeights(factor_returns)
            
            # Select the weighting method
            if weight_method == 'equal':
                weights = np.ones(factor_loadings.shape[1]) / factor_loadings.shape[1]
            elif weight_method == 'inverse_volatility':
                inv_vols = 1 / factor_volatility
                weights = inv_vols / np.sum(inv_vols)
            elif weight_method == 'risk_parity':
                weights = portfolio_weights.risk_parity_weights()
            elif weight_method == 'momentum':
                weights = portfolio_weights.momentum_based_weights()
            elif weight_method == 'tail_risk_parity':
                weights = portfolio_weights.tail_risk_parity_weights(alpha=0.05)
            elif weight_method == 'random_forest':
                weights = portfolio_weights.random_forest_weights()
            else:
                raise ValueError(f"Unknown weight method: {weight_method}")

            train_factor_returns.append(factor_returns)
            train_factor_loadings.append(factor_loadings)
            test_index.append(test_returns.index[0])

            for i in range(factor_loadings.shape[1]):
                weighted_average_factor_returns = np.zeros(factor_loadings.shape[0])
                for j in range(factor_returns.shape[1]):
                    weighted_average_factor_returns += factor_loadings[:, i] * factor_returns[i, j] / self.window_size
                asset_ranks = np.argsort(np.argsort(weighted_average_factor_returns))
                top_quintile = asset_ranks >= (len(asset_ranks) * 0.90)
                bottom_quintile = asset_ranks <= (len(asset_ranks) * 0.10)
                long_weights = np.ones(np.sum(top_quintile)) / np.sum(top_quintile)
                short_weights = np.ones(np.sum(bottom_quintile)) / np.sum(bottom_quintile)
                long_assets = test_returns.iloc[:, top_quintile]
                short_assets = test_returns.iloc[:, bottom_quintile]
                long_return += np.dot(long_assets.values.flatten(), long_weights) * weights[i]
                short_return += np.dot(short_assets.values.flatten(), short_weights) * weights[i]

            portfolio_return = long_return - short_return

            # Adjust for transaction costs and slippage
            transaction_costs = self.transaction_cost * capital
            slippage_costs = self.slippage * capital
            net_portfolio_return = (1 + portfolio_return) * capital - transaction_costs - slippage_costs

            # Update capital
            capital = net_portfolio_return

            portfolio_capital.append(capital)

        portfolio_capital_series = pd.Series(portfolio_capital, index=test_index)
        return portfolio_capital_series

    def evaluate_strategies(self, index_returns):
        index_returns_series = pd.Series(index_returns, index=self.data_returns.index[self.window_size:])
        portfolio_returns_dict = {}

        for method in self.weight_methods:
            self.portfolio_capital_dict[method] = self.rolling_apca_strategy(weight_method=method)
            portfolio_returns_dict[method] = self.portfolio_capital_dict[method].pct_change().dropna()

        plt.figure(figsize=(12, 6))
        for method, capital in self.portfolio_capital_dict.items():
            cumulative_capital = (capital / self.initial_capital)
            plt.plot(cumulative_capital, label=f'Portfolio Capital ({method})')
        cumulative_index_capital = (index_returns_series * self.initial_capital + self.initial_capital) / self.initial_capital
        plt.plot(cumulative_index_capital, label='Index Capital', linewidth=2, linestyle='--')
        plt.xticks(rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.title('Capital Appreciation of Portfolio vs Index')
        plt.legend()
        plt.grid(True)
        plt.show()

        return index_returns_series, portfolio_returns_dict


class FinancialMetrics:
    @staticmethod
    def _annualize(metric_func, r, periods_per_year, **kwargs):
        result = r.aggregate(metric_func, periods_per_year=periods_per_year, **kwargs)
        return result

    @staticmethod
    def _risk_free_adjusted_returns(r, riskfree_rate, periods_per_year):
        rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
        return r - rf_per_period

    @staticmethod
    def drawdown(return_series: pd.Series):
        """
        Takes a time series of asset returns
        Computes and returns a data frame that contains:
        the wealth index, the previous peaks, and percent drawdowns
        """
        wealth_index = 1000 * (1 + return_series).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdown = (wealth_index - previous_peaks) / previous_peaks
        return pd.DataFrame({
            "Wealth": wealth_index,
            "Peaks": previous_peaks,
            "Drawdown": drawdown
            })

    @staticmethod
    def semideviation(r, periods_per_year):
        """
        Compute the Annualized Semi-Deviation
        """
        neg_rets = r[r < 0]
        return FinancialMetrics.annualize_vol(r=neg_rets, periods_per_year=periods_per_year)

    @staticmethod
    def skewness(r):
        """
        Computes the skewness of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**3).mean()
        return exp / sigma_r**3

    @staticmethod
    def kurtosis(r):
        """
        Computes the kurtosis of the supplied Series or DataFrame
        Returns a float or a Series
        """
        demeaned_r = r - r.mean()
        sigma_r = r.std(ddof=0)
        exp = (demeaned_r**4).mean()
        return exp / sigma_r**4

    @staticmethod
    def var_historic(r, level=5):
        """
        VaR Historic
        """
        if isinstance(r, pd.DataFrame):
            return r.aggregate(FinancialMetrics.var_historic, level=level)
        elif isinstance(r, pd.Series):
            return -np.percentile(r, level)
        else:
            raise TypeError("Expected r to be Series or DataFrame")

    @staticmethod
    def var_gaussian(r, level=5, modified=False):
        """
        Returns the Parametric Gaussian VaR of a Series or DataFrame
        If "modified" is True, then the modified VaR is returned,
        using the Cornish-Fisher modification
        """
        z = norm.ppf(level / 100)
        if modified:
            s = FinancialMetrics.skewness(r)
            k = FinancialMetrics.kurtosis(r)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3 * z) * (k - 3) / 24 -
                 (2 * z**3 - 5 * z) * (s**2) / 36)
        return -(r.mean() + z * r.std(ddof=0))

    @staticmethod
    def cvar_historic(r, level=5):
        """
        Computes the Conditional VaR of Series or DataFrame
        """
        if isinstance(r, pd.Series):
            is_beyond = r <= -FinancialMetrics.var_historic(r, level=level)
            return -r[is_beyond].mean()
        elif isinstance(r, pd.DataFrame):
            return r.aggregate(FinancialMetrics.cvar_historic, level=level)
        else:
            raise TypeError("Expected r to be a Series or DataFrame")

    @staticmethod
    def annualize_rets(r, periods_per_year):
        """
        Annualizes a set of returns
        """
        compounded_growth = (1 + r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(periods_per_year / n_periods) - 1

    @staticmethod
    def annualize_vol(r, periods_per_year):
        """
        Annualizes the vol of a set of returns
        """
        return r.std() * (periods_per_year**0.5)

    @staticmethod
    def sharpe_ratio(r, riskfree_rate, periods_per_year):
        """
        Computes the annualized Sharpe ratio of a set of returns
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        ann_vol = FinancialMetrics.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def rovar(r, periods_per_year, level=5):
        """
        Compute the Return on Value-at-Risk
        """
        return FinancialMetrics.annualize_rets(r, periods_per_year=periods_per_year) / abs(FinancialMetrics.var_historic(r, level=level)) if abs(FinancialMetrics.var_historic(r, level=level)) > 1e-10 else 0

    @staticmethod
    def sortino_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Sortino Ratio of a set of returns
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        neg_rets = excess_ret[excess_ret < 0]
        ann_vol = FinancialMetrics.annualize_vol(neg_rets, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def calmar_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Calmar Ratio of a set of returns
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        max_dd = abs(FinancialMetrics.drawdown(r).Drawdown.min())
        return ann_ex_ret / max_dd if max_dd != 0 else 0

    @staticmethod
    def burke_ratio(r, riskfree_rate, periods_per_year, modified=False):
        """
        Compute the annualized Burke Ratio of a set of returns
        If "modified" is True, then the modified Burke Ratio is returned
        """
        excess_ret = FinancialMetrics._risk_free_adjusted_returns(r, riskfree_rate, periods_per_year)
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        sum_dwn = np.sqrt(np.sum((FinancialMetrics.drawdown(r).Drawdown)**2))
        if not modified:
            bk_ratio = ann_ex_ret / sum_dwn if sum_dwn != 0 else 0
        else:
            bk_ratio = ann_ex_ret / sum_dwn * np.sqrt(len(r)) if sum_dwn != 0 else 0
        return bk_ratio

    @staticmethod
    def net_profit(returns):
        """
        Calculates the net profit of a strategy.
        """
        cumulative_returns = (1 + returns).cumprod() - 1
        return cumulative_returns.iloc[-1]

    @staticmethod
    def worst_drawdown(returns):
        """
        Calculates the worst drawdown from cumulative returns.
        """
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    @staticmethod
    def summary_stats(r, riskfree_rate=0.03, periods_per_year=12):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = FinancialMetrics._annualize(FinancialMetrics.annualize_rets, r, periods_per_year)
        ann_vol = FinancialMetrics._annualize(FinancialMetrics.annualize_vol, r, periods_per_year)
        semidev = FinancialMetrics._annualize(FinancialMetrics.semideviation, r, periods_per_year)
        ann_sr = FinancialMetrics._annualize(FinancialMetrics.sharpe_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_cr = FinancialMetrics._annualize(FinancialMetrics.calmar_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_br = FinancialMetrics._annualize(FinancialMetrics.burke_ratio, r, periods_per_year, riskfree_rate=riskfree_rate, modified=True)
        ann_sortr = FinancialMetrics._annualize(FinancialMetrics.sortino_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        dd = r.aggregate(lambda r: FinancialMetrics.drawdown(r).Drawdown.min())
        skew = r.aggregate(FinancialMetrics.skewness)
        kurt = r.aggregate(FinancialMetrics.kurtosis)
        hist_var5 = r.aggregate(FinancialMetrics.var_historic)
        cf_var5 = r.aggregate(FinancialMetrics.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(FinancialMetrics.cvar_historic)
        rovar5 = r.aggregate(FinancialMetrics.rovar, periods_per_year=periods_per_year)
        np_wdd_ratio = r.aggregate(lambda returns: FinancialMetrics.net_profit(returns) / -FinancialMetrics.worst_drawdown(returns))
        return pd.DataFrame({
            "Annualized Return": round(ann_r, 4),
            "Annualized Volatility": round(ann_vol, 4),
            "Semi-Deviation": round(semidev, 4),
            "Skewness": round(skew, 4),
            "Kurtosis": round(kurt, 4),
            "Historic VaR (5%)": round(hist_var5, 4),
            "Cornish-Fisher VaR (5%)": round(cf_var5, 4),
            "Historic CVaR (5%)": round(hist_cvar5, 4),
            "Return on VaR": round(rovar5, 4),
            "Sharpe Ratio": round(ann_sr, 4),
            "Sortino Ratio": round(ann_sortr, 4),
            "Calmar Ratio": round(ann_cr, 4),
            "Modified Burke Ratio": round(ann_br, 4),
            "Max Drawdown": round(dd, 4),
            "Net Profit to Worst Drawdown": round(np_wdd_ratio, 4)
        })