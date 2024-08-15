import pywt
import numpy as np
from dtaidistance import dtw
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.regression.quantile_regression import QuantReg


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
        return weights / np.sum(weights)

    def random_forest_weights(self, random_seed=42):
        np.random.seed(random_seed)
        X = self.factor_returns[:-1]
        y = self.factor_returns[1:]
        tscv = TimeSeriesSplit(n_splits=3)
        val_mse_list = []
        train_mse_list = []
        n_estimators_grid = [10, 20, 30, 40, 50]
        max_depth_grid = [3, 5, 10, 20]
        min_samples_split_grid = [2, 3, 5]
        best_val_mse = float("inf")
        best_params = {}

        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                for min_samples_split in min_samples_split_grid:
                    val_mse_list = []
                    train_mse_list = []
                    for train_index, val_index in tscv.split(X):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
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
                        best_train_mse = train_mse_list
                        best_val_mse_per_split = val_mse_list
                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                        }

        best_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=random_seed,
        )
        best_model.fit(X, y)
        weights = best_model.feature_importances_
        weights = weights / np.sum(weights)
        return weights, best_params, best_train_mse, best_val_mse_per_split

    def extra_trees_weights(self, random_seed=42):
        np.random.seed(random_seed)
        X = self.factor_returns[:-1]
        y = self.factor_returns[1:]
        tscv = TimeSeriesSplit(n_splits=3)
        n_estimators_grid = [10, 20, 30, 40, 50]
        max_depth_grid = [3, 5, 10, 20]
        min_samples_split_grid = [2, 3, 5]
        best_val_mse = float("inf")
        best_params = {}

        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                for min_samples_split in min_samples_split_grid:
                    val_mse_list = []
                    train_mse_list = []
                    for train_index, val_index in tscv.split(X):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]
                        model = ExtraTreesRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
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
                        best_train_mse = train_mse_list
                        best_val_mse_per_split = val_mse_list
                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                        }

        best_model = ExtraTreesRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=random_seed,
        )
        best_model.fit(X, y)
        weights = best_model.feature_importances_
        weights = weights / np.sum(weights)
        return weights, best_params, best_train_mse, best_val_mse_per_split
    
    def gradient_boosting_weights(self, random_seed=42):
        np.random.seed(random_seed)
        X = self.factor_returns[:-1]
        y = self.factor_returns[1:]
        tscv = TimeSeriesSplit(n_splits=3)
        val_mse_list = []
        train_mse_list = []
        n_estimators_grid = [10, 20, 30, 40, 50]
        max_depth_grid = [3, 5, 10, 20]
        min_samples_split_grid = [2, 3, 5]
        learning_rate_grid = [0.01, 0.05, 0.1, 0.2]
        best_val_mse = float("inf")
        best_params = {}

        for learning_rate in learning_rate_grid:
            for n_estimators in n_estimators_grid:
                for max_depth in max_depth_grid:
                    for min_samples_split in min_samples_split_grid:
                        val_mse_list = []
                        train_mse_list = []
                        for train_index, val_index in tscv.split(X):
                            X_train, X_val = X[train_index], X[val_index]
                            y_train, y_val = y[train_index], y[val_index]
                            model = MultiOutputRegressor(GradientBoostingRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                learning_rate=learning_rate,
                                random_state=random_seed
                            ))
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
                            best_train_mse = train_mse_list
                            best_val_mse_per_split = val_mse_list
                            best_params = {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "min_samples_split": min_samples_split,
                                "learning_rate": learning_rate
                            }

        best_model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            learning_rate=best_params["learning_rate"],
            random_state=random_seed
        ))
        best_model.fit(X, y)
        weights = np.mean([estimator.feature_importances_ for estimator in best_model.estimators_], axis=0)
        weights = weights / np.sum(weights)
        return weights, best_params, best_train_mse, best_val_mse_per_split