import json
import math
import pywt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from dtaidistance import dtw
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.regression.quantile_regression import QuantReg


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
        with open("Tickers.json", "r") as json_file:
            tickers_json = json.load(json_file)
        return tickers_json["tickers"]

    def download_data(self, tickers):
        return yf.download(
            tickers, start=self.start_date, end=self.end_date, progress=False
        )["Open"]

    def download_volume_data(self, tickers):
        return yf.download(
            tickers, start=self.start_date, end=self.end_date, progress=False
        )["Volume"]

    def download_index_data(self, index_ticker):
        return yf.download(
            index_ticker, start=self.start_date, end=self.end_date, progress=False
        )["Open"]

    def resample_data(self, data):
        return data.resample("ME").first()

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
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            fmt=".1f",
        )
        plt.title("Correlation Heatmap of Returns")
        plt.show()


class ComponentsAnalysis:
    def __init__(self, returns):
        self.returns = returns
        self.R = returns.values
        self.t, self.n = returns.shape
        self.Q_tilde = self.calculate_covariance_matrix()
        self.eigenvalues, self.eigenvectors = self.perform_eigendecomposition()
        self.explained_variance_ratio = self.calculate_explained_variance_ratio()
        self.cumulative_explained_variance = (
            self.calculate_cumulative_explained_variance()
        )

    def calculate_covariance_matrix(self):
        return (
            (1 / self.t) * self.R @ self.R.T
        )  # asset returns covariance matrix (in t x t not n x n, nello screenshot assume R che ha t come colonne)

    def perform_eigendecomposition(self):
        return np.linalg.eigh(self.Q_tilde)

    def calculate_explained_variance_ratio(self):
        return self.eigenvalues[::-1] / np.sum(self.eigenvalues)

    def calculate_cumulative_explained_variance(self):
        return np.cumsum(self.explained_variance_ratio)

    def plot_scree(self):
        plt.figure(figsize=(20, 6))
        plt.plot(
            np.arange(1, len(self.eigenvalues) + 1),
            self.eigenvalues[::-1],
            "o-",
            markersize=8,
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Eigenvalue")
        plt.title("Scree Plot")
        plt.grid(True)
        plt.show()

    def plot_cumulative_explained_variance(self):
        plt.figure(figsize=(20, 6))
        plt.plot(
            np.arange(1, len(self.eigenvalues) + 1),
            self.cumulative_explained_variance,
            "o-",
            markersize=8,
        )
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.axhline(y=0.90, color="r", linestyle="--")
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
        self.cumulative_explained_variance = (
            self.calculate_cumulative_explained_variance()
        )
        self.m = self.number_factors(0.90)
        self.U_m_final, self.F_final, self.B_final, self.Gamma_final = (
            self.iterative_estimation()
        )

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
            U_m = self.eigenvectors[:, -self.m :]
            F = U_m.T  # Factor returns
            B = self.R.T @ U_m  # Factor exposures
            Gamma = self.R.T - B @ F  # Specific returns
            Delta_squared = (1 / self.t) * np.diag(
                Gamma @ Gamma.T
            )  # Specific covariance matrix

            if np.all(
                np.abs(Delta_squared - previous_Delta_squared)
                < self.convergence_threshold
            ):
                # print(f"Converged after {iteration + 1} iterations")
                break

            previous_Delta_squared = Delta_squared
            Delta_inv = np.diag(1 / np.sqrt(Delta_squared))
            R_star = Delta_inv @ self.R.T
            Q_tilde_star = (1 / self.n) * R_star.T @ R_star
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(Q_tilde_star)

        else:
            print("Did not converge within the maximum number of iterations")

        U_m_final = self.eigenvectors[:, -self.m :]
        F_final = U_m_final.T
        B_final = self.R.T @ U_m_final
        Gamma_final = self.R.T - B_final @ F_final

        return U_m_final, F_final, B_final, Gamma_final


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

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )

        return result.x

    def momentum_based_weights(self, lookback_period=4):
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
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )

        return result.x
    
    def wavelet_weights(self, wavelet='haar'):
        coeffs = pywt.dwt(self.factor_returns.T, wavelet)
        approx, _ = coeffs
        if len(approx.shape) > 1:
            approx = np.mean(approx, axis=0)
        weights = approx / np.sum(approx)
        return weights

    def minimum_correlation_weights(self):
        corr_matrix = np.corrcoef(self.factor_returns.T)
        initial_weights = np.ones(len(corr_matrix)) / len(corr_matrix)

        def objective(weights):
            weighted_corr = weights.T @ corr_matrix @ weights
            return weighted_corr

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )
        return result.x

    def minimum_variance_weights(self):
        cov_matrix = np.cov(self.factor_returns.T)

        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            return portfolio_vol

        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )
        return result.x

    def spearman_ic_weights(self, lookback_period=12):
        rank_returns = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 0, self.factor_returns[-lookback_period:])
        rank_avg_returns = np.mean(rank_returns, axis=0)
        positive_rank_avg = rank_avg_returns.clip(min=0)
        weights = positive_rank_avg / np.sum(positive_rank_avg)
        return weights

    def maximum_diversification_weights(self):
        cov_matrix = np.cov(self.factor_returns.T)
        volatilities = np.sqrt(np.diag(cov_matrix))

        def objective(weights):
            weighted_volatility = np.dot(weights, volatilities)
            diversification_ratio = weighted_volatility / np.sqrt(weights.T @ cov_matrix @ weights)
            return -diversification_ratio

        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )
        return result.x

    def maximum_sharpe_ratio_weights(self, risk_free_rate=0):
        expected_returns = np.mean(self.factor_returns, axis=0)
        cov_matrix = np.cov(self.factor_returns.T)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio

        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )
        return result.x

    def algo_complexity_weights(self):
        def complexity_function(weights):
            return np.count_nonzero(weights)

        cov_matrix = np.cov(self.factor_returns.T)
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)

        def objective(weights):
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            complexity_penalty = complexity_function(weights)
            return portfolio_vol + complexity_penalty

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(initial_weights))]
        result = minimize(
            objective, initial_weights, bounds=bounds, constraints=constraints
        )
        return result.x
    
    def entropy_based_weights(self):
        cov_matrix = np.cov(self.factor_returns.T)
        
        def entropy(weights):
            return -np.sum(weights * np.log(weights))

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(cov_matrix))]
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        result = minimize(entropy, initial_weights, bounds=bounds, constraints=constraints)
        return result.x

    def kelly_criterion_weights(self):
        mean_returns = np.mean(self.factor_returns, axis=0)
        cov_matrix = np.cov(self.factor_returns.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        weights = inv_cov_matrix @ mean_returns
        return weights / np.sum(weights)

    def maximin_weights(self):
        cov_matrix = np.cov(self.factor_returns.T)
        
        def objective(weights):
            return -np.min(weights.T @ cov_matrix @ weights)
        
        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(cov_matrix))]
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        return result.x

    def mean_absolute_deviation_weights(self):
        abs_dev = np.mean(np.abs(self.factor_returns - np.mean(self.factor_returns, axis=0)), axis=0)
        weights = 1 / abs_dev
        return weights / np.sum(weights)

    def quantile_regression_weights(self, quantile=0.5):
        mean_returns = np.mean(self.factor_returns, axis=0)
        if self.factor_returns.shape[0] != len(mean_returns):
            mean_returns = np.mean(self.factor_returns, axis=1)
        model = QuantReg(mean_returns, self.factor_returns)
        res = model.fit(q=quantile)
        weights = res.params
        return weights / np.sum(weights)

    def dtw_weights(self):
        distances = np.array([dtw.distance(self.factor_returns[:, i], np.mean(self.factor_returns, axis=1)) for i in range(self.factor_returns.shape[1])])
        weights = 1 / distances
        return weights / np.sum(weights)
    
    def convex_optimization_weights(self):
        cov_matrix = np.cov(self.factor_returns.T)

        def objective(weights):
            return np.sqrt(weights.T @ cov_matrix @ weights)

        constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]
        bounds = [(0, 1) for _ in range(len(cov_matrix))]
        initial_weights = np.ones(len(cov_matrix)) / len(cov_matrix)
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        return result.x

    def stochastic_dominance_weights(self, order=1):
        sorted_returns = np.sort(self.factor_returns, axis=0)
        cumulative_returns = np.cumsum(sorted_returns, axis=0)
        weights = np.mean(cumulative_returns, axis=0) ** order
        return weights / np.sum(weights)  # Normalize to sum to 1

    def random_forest_weights(self, random_seed=42):
        np.random.seed(random_seed)
        X = self.factor_returns[:-1]
        y = self.factor_returns[1:]
        tscv = TimeSeriesSplit(n_splits=5)
        val_mse_list = []
        train_mse_list = []
        n_estimators_grid = [5, 10, 20, 30, 40, 50]
        max_depth_grid = [5, 10, 20, 30, 40, 50]
        #n_estimators_grid = [50]
        #max_depth_grid = [1]
        best_val_mse = float("inf")
        best_params = {}

        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                #for min_samples_split in min_samples_split_grid:
                val_mse_list = []
                train_mse_list = []
                for train_index, val_index in tscv.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]
                    model = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        #min_samples_split=min_samples_split,
                        random_state=random_seed,
                    )
                    model.fit(X_train, y_train)
                    train_predictions = model.predict(X_train)
                    val_predictions = model.predict(X_val)
                    train_mse = mean_squared_error(y_train, train_predictions)
                    val_mse = mean_squared_error(y_val, val_predictions)
                    train_mse_list.append(train_mse)
                    val_mse_list.append(val_mse)
                mean_val_mse = np.mean(val_mse_list)
                if mean_val_mse < best_val_mse:
                    best_val_mse = mean_val_mse
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        #"min_samples_split": min_samples_split,
                    }

        best_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            #min_samples_split=best_params["min_samples_split"],
            random_state=random_seed,
        )
        best_model.fit(X, y)

        # Use feature importances as weights
        weights = best_model.feature_importances_
        # Increase spread: Scale weights by a factor (e.g., square root) to reduce dominance
        #weights = np.power(weights, 2)
        weights = weights / np.sum(weights)
        return weights


class RollingAPCAStrategy:
    def __init__(
        self,
        data_returns,
        window_size,
        max_iterations,
        transaction_cost=0.001,
        slippage=0.001,
    ):
        self.data_returns = data_returns
        self.window_size = window_size
        self.max_iterations = max_iterations
        self.weight_methods = [
            "equal",
            "risk_parity",
            "momentum",
            #"tail_risk_parity",
            #"wavelet",
            #"min_corr",
            #"min_var",
            #"spearman_ic",
            #"max_div",
            #"max_sr",
            #"algo",
            #"entropy",
            #"kelly", 
            #UNCOMMENTARE PERCHE RITORNO ASSURDO, BUONO SHARPE MA DRAWDOWN AL 48%
            "maximin",
            #"mad",
            #"quantile",
            #"dtw",
            "stochastic",
            #"convex"
            #"random_forest"
        ]
        self.portfolio_returns_dict = {}
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def rolling_apca_strategy(self, weight_method):
        test_index = []
        portfolio_returns = []

        for start in range(len(self.data_returns) - self.window_size):
            long_return = 0
            short_return = 0
            end = start + self.window_size
            train_returns = self.data_returns.iloc[start:end]  # t x n
            test_returns = self.data_returns.iloc[end : end + 1]
            test_index.append(test_returns.index[0])
            factor_model = APCA(train_returns, max_iterations=self.max_iterations)
            factor_returns = factor_model.F_final  # m x t
            factor_loadings = factor_model.B_final  # n x m
            #factor_specific = factor_model.Gamma_final  # n x t
            portfolio_weights = PortfolioWeights(factor_returns.T)
            
            # Select the weighting method
            if weight_method == "equal":
                weights = np.ones(factor_loadings.shape[1]) / factor_loadings.shape[1]
            elif weight_method == "risk_parity":
                weights = portfolio_weights.risk_parity_weights()
            elif weight_method == "momentum":
                weights = portfolio_weights.momentum_based_weights()
            elif weight_method == "tail_risk_parity":
                weights = portfolio_weights.tail_risk_parity_weights(alpha=0.05)
            elif weight_method == "wavelet":
                weights = portfolio_weights.wavelet_weights()
            elif weight_method == "min_corr":
                weights = portfolio_weights.minimum_correlation_weights()
            elif weight_method == "min_var":
                weights = portfolio_weights.minimum_variance_weights()
            elif weight_method == "spearman_ic":
                weights = portfolio_weights.spearman_ic_weights()
            elif weight_method == "max_div":
                weights = portfolio_weights.maximum_diversification_weights()
            elif weight_method == "max_sr":
                weights = portfolio_weights.maximum_sharpe_ratio_weights()
            elif weight_method == "algo":
                weights = portfolio_weights.algo_complexity_weights()
            elif weight_method == "entropy":
                weights = portfolio_weights.entropy_based_weights()
            elif weight_method == "kelly":
                weights = portfolio_weights.kelly_criterion_weights()
            elif weight_method == "maximin":
                weights = portfolio_weights.maximin_weights()
            elif weight_method == "mad":
                weights = portfolio_weights.mean_absolute_deviation_weights()
            elif weight_method == "quantile":
                weights = portfolio_weights.quantile_regression_weights()
            elif weight_method == "dtw":
                weights = portfolio_weights.dtw_weights()
            elif weight_method == "stochastic":
                weights = portfolio_weights.stochastic_dominance_weights()
            elif weight_method == "convex":
                weights = portfolio_weights.convex_optimization_weights()
            elif weight_method == "random_forest":
                weights = portfolio_weights.random_forest_weights()
            else:
                raise ValueError(f"Unknown weight method: {weight_method}")

            for i in range(factor_loadings.shape[1]):
                weighted_average_factor_returns = np.zeros(factor_loadings.shape[0])
                for j in range(factor_returns.shape[1]):
                    weighted_average_factor_returns += (factor_loadings[:, i] * factor_returns[i, j])
                weighted_average_factor_returns /= self.window_size
                asset_ranks = np.argsort(np.argsort(weighted_average_factor_returns))
                top_quintile = asset_ranks >= (len(asset_ranks) * 0.90)
                bottom_quintile = asset_ranks <= (len(asset_ranks) * 0.10)
                long_weights = np.ones(np.sum(top_quintile)) / np.sum(top_quintile)
                short_weights = np.ones(np.sum(bottom_quintile)) / np.sum(bottom_quintile)
                long_assets = test_returns.iloc[:, top_quintile]
                short_assets = test_returns.iloc[:, bottom_quintile]
                long_return += (np.dot(long_assets.values.flatten(), long_weights) * weights[i])
                short_return += (np.dot(short_assets.values.flatten(), short_weights) * weights[i])
                
            net_portfolio_return = (long_return - short_return) - self.transaction_cost - self.slippage
            portfolio_returns.append(net_portfolio_return)

        portfolio_returns_series = pd.Series(portfolio_returns, index=test_index)
        return portfolio_returns_series

    def evaluate_strategies(self, index_returns):
        index_returns_series = pd.Series(
            index_returns, index=self.data_returns.index[self.window_size :]
        )
        portfolio_returns_dict = {}

        for method in self.weight_methods:
            self.portfolio_returns_dict[method] = self.rolling_apca_strategy(
                weight_method=method
            )
            portfolio_returns_dict[method] = self.portfolio_returns_dict[
                method
            ].dropna()

        plt.figure(figsize=(12, 6))
        for method, returns in self.portfolio_returns_dict.items():
            cumulative_returns = (1 + returns).cumprod()
            cumulative_returns = pd.Series(
                [1] + cumulative_returns.tolist(),
                index=[self.data_returns.index[self.window_size - 1]]
                + cumulative_returns.index.tolist(),
            )
            plt.plot(cumulative_returns, label=f"Portfolio Returns ({method})")

        cumulative_index_returns = (1 + index_returns_series).cumprod()
        cumulative_index_returns = pd.Series(
            [1] + cumulative_index_returns.tolist(),
            index=[self.data_returns.index[self.window_size - 1]]
            + cumulative_index_returns.index.tolist(),
        )
        plt.plot(
            cumulative_index_returns, label="Index Returns", linewidth=2, linestyle="--"
        )
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns")
        plt.title("Cumulative Returns of Portfolio vs Index")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot Drawdowns
        plt.figure(figsize=(12, 6))
        for method, returns in portfolio_returns_dict.items():
            drawdown = FinancialMetrics.drawdown(returns)["Drawdown"]
            plt.plot(drawdown, label=f"Portfolio Drawdown ({method})")
        index_drawdown = FinancialMetrics.drawdown(index_returns_series)["Drawdown"]
        plt.plot(index_drawdown, label="Index Drawdown", linewidth=2, linestyle="--")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.title("Drawdowns of Portfolio vs Index")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot heatmap of monthly returns
        def monthly_returns_heatmap(returns, title, ax):
            returns_df = returns.to_frame(name="Returns")
            returns_df.index = pd.to_datetime(returns_df.index)
            returns_df["Year"] = returns_df.index.year
            returns_df["Month"] = returns_df.index.month
            monthly_returns = returns_df.pivot_table(
                index="Year", columns="Month", values="Returns", aggfunc="sum"
            )

            sns.heatmap(
                monthly_returns,
                annot=True,
                fmt=".2%",
                cmap="RdYlGn",
                center=0,
                cbar=False,
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("Month")
            ax.set_ylabel("Year")

        num_methods = len(portfolio_returns_dict)
        num_cols = 2
        num_rows = math.ceil(num_methods / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 8 * num_rows))
        axes = axes.flatten()

        for ax, (method, returns) in zip(axes, portfolio_returns_dict.items()):
            monthly_returns_heatmap(
                returns, title=f"Monthly Returns for Portfolio ({method})", ax=ax
            )

        for i in range(len(portfolio_returns_dict), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

        return index_returns_series, portfolio_returns_dict


class FinancialMetrics:
    @staticmethod
    def annualize(metric_func, r, periods_per_year, **kwargs):
        result = r.aggregate(metric_func, periods_per_year=periods_per_year, **kwargs)
        return result

    @staticmethod
    def risk_free_adjusted_returns(r, riskfree_rate, periods_per_year):
        rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
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
        return pd.DataFrame(
            {"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdown}
        )

    @staticmethod
    def semideviation(r, periods_per_year):
        """
        Compute the Annualized Semi-Deviation
        """
        neg_rets = r[r < 0]
        return FinancialMetrics.annualize_vol(
            r=neg_rets, periods_per_year=periods_per_year
        )

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
            z = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * (k - 3) / 24
                - (2 * z**3 - 5 * z) * (s**2) / 36
            )
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
        if compounded_growth <= 0:
            return 0

        return compounded_growth ** (periods_per_year / n_periods) - 1

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
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        ann_vol = FinancialMetrics.annualize_vol(r, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def rovar(r, periods_per_year, level=5):
        """
        Compute the Return on Value-at-Risk
        """
        return (
            FinancialMetrics.annualize_rets(r, periods_per_year=periods_per_year)
            / abs(FinancialMetrics.var_historic(r, level=level))
            if abs(FinancialMetrics.var_historic(r, level=level)) > 1e-10
            else 0
        )

    @staticmethod
    def sortino_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Sortino Ratio of a set of returns
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        neg_rets = excess_ret[excess_ret < 0]
        ann_vol = FinancialMetrics.annualize_vol(neg_rets, periods_per_year)
        return ann_ex_ret / ann_vol if ann_vol != 0 else 0

    @staticmethod
    def calmar_ratio(r, riskfree_rate, periods_per_year):
        """
        Compute the annualized Calmar Ratio of a set of returns
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        max_dd = abs(FinancialMetrics.drawdown(r).Drawdown.min())
        return ann_ex_ret / max_dd if max_dd != 0 else 0

    @staticmethod
    def burke_ratio(r, riskfree_rate, periods_per_year, modified=False):
        """
        Compute the annualized Burke Ratio of a set of returns
        If "modified" is True, then the modified Burke Ratio is returned
        """
        excess_ret = FinancialMetrics.risk_free_adjusted_returns(
            r, riskfree_rate, periods_per_year
        )
        ann_ex_ret = FinancialMetrics.annualize_rets(excess_ret, periods_per_year)
        sum_dwn = np.sqrt(np.sum((FinancialMetrics.drawdown(r).Drawdown) ** 2))
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
    def tracking_error(r, market):
        difference = r - market
        return difference.std()

    @staticmethod
    def information_ratio(r, market, periods_per_year):
        diff_rets = r - market
        ann_diff_rets = FinancialMetrics.annualize_rets(diff_rets, periods_per_year)
        tracking_err = FinancialMetrics.tracking_error(r, market)
        return ann_diff_rets / tracking_err if tracking_err != 0 else 0

    @staticmethod
    def tail_ratio(r):
        tail_ratio = np.percentile(r, 95) / abs(np.percentile(r, 5))
        return tail_ratio

    @staticmethod
    def summary_stats(r, market=None, riskfree_rate=0.03, periods_per_year=12):
        """
        Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
        """
        ann_r = FinancialMetrics.annualize(FinancialMetrics.annualize_rets, r, periods_per_year)
        ann_vol = FinancialMetrics.annualize(FinancialMetrics.annualize_vol, r, periods_per_year)
        semidev = FinancialMetrics.annualize(FinancialMetrics.semideviation, r, periods_per_year)
        ann_sr = FinancialMetrics.annualize(FinancialMetrics.sharpe_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_cr = FinancialMetrics.annualize(FinancialMetrics.calmar_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        ann_br = FinancialMetrics.annualize(FinancialMetrics.burke_ratio, r, periods_per_year, riskfree_rate=riskfree_rate, modified=True)
        ann_sortr = FinancialMetrics.annualize(FinancialMetrics.sortino_ratio, r, periods_per_year, riskfree_rate=riskfree_rate)
        dd = r.aggregate(lambda r: FinancialMetrics.drawdown(r).Drawdown.min())
        skew = r.aggregate(FinancialMetrics.skewness)
        kurt = r.aggregate(FinancialMetrics.kurtosis)
        hist_var5 = r.aggregate(FinancialMetrics.var_historic)
        cf_var5 = r.aggregate(FinancialMetrics.var_gaussian, modified=True)
        hist_cvar5 = r.aggregate(FinancialMetrics.cvar_historic)
        rovar5 = r.aggregate(FinancialMetrics.rovar, periods_per_year=periods_per_year)
        np_wdd_ratio = r.aggregate(lambda returns: FinancialMetrics.net_profit(returns) / -FinancialMetrics.worst_drawdown(returns))
        tail_ratio = r.aggregate(FinancialMetrics.tail_ratio)
        if market is not None:
            market_series = market.squeeze()
            tracking_err = r.aggregate(FinancialMetrics.tracking_error, market=market_series)
            info_ratio = r.aggregate(FinancialMetrics.information_ratio, market=market_series, periods_per_year=periods_per_year)
        else:
            tracking_err = pd.Series(0 * len(r.columns), index=r.columns)
            info_ratio = pd.Series(0 * len(r.columns), index=r.columns)

        return pd.DataFrame(
            {
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
                "Net Profit to Worst Drawdown": round(np_wdd_ratio, 4),
                "Tracking Error": round(tracking_err, 4),
                "Information Ratio": round(info_ratio, 4),
                "Tail Ratio": round(tail_ratio, 4)
            }
        )
