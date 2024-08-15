import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import models.APCA as APCA
import models.PortfolioWeights as PortfolioWeights
import models.FinancialMetrics as FinancialMetrics

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
            "maximin",
            #"mad",
            #"quantile",
            #"dtw",
            "stochastic",
            #"convex"
            #"random_forest",
            "gradient_boosting",
            #"extra_trees",
        ]
        self.portfolio_returns_dict = {}
        self.transaction_cost = transaction_cost
        self.slippage = slippage

    def rolling_apca_strategy(self, weight_method):
        test_index = []
        portfolio_returns = []
        best_params_list = []
        best_train_mse_list = []
        best_val_mse_list = []

        for start in range(len(self.data_returns) - self.window_size):
            long_return = 0
            short_return = 0
            end = start + self.window_size
            train_returns = self.data_returns.iloc[start:end]  # t x n
            test_returns = self.data_returns.iloc[end : end + 1]
            test_index.append(test_returns.index[0])
            factor_model = APCA.APCA(train_returns, max_iterations=self.max_iterations)
            factor_returns = factor_model.F_final  # m x t
            factor_loadings = factor_model.B_final  # n x m
            portfolio_weights = PortfolioWeights.PortfolioWeights(factor_returns.T)
            
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
                weights, best_params, best_train_mse, best_val_mse_per_split = portfolio_weights.random_forest_weights()
                best_params_list.append(best_params)
                best_train_mse_list.extend(best_train_mse)
                best_val_mse_list.extend(best_val_mse_per_split)
            elif weight_method == "gradient_boosting":
                weights, best_params, best_train_mse, best_val_mse_per_split = portfolio_weights.gradient_boosting_weights()
                best_params_list.append(best_params)
                best_train_mse_list.extend(best_train_mse)
                best_val_mse_list.extend(best_val_mse_per_split)
            elif weight_method == "extra_trees":
                weights, best_params, best_train_mse, best_val_mse_per_split = portfolio_weights.extra_trees_weights()
                best_params_list.append(best_params)
                best_train_mse_list.extend(best_train_mse)
                best_val_mse_list.extend(best_val_mse_per_split)
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

        if weight_method in ["random_forest", "gradient_boosting", "extra_trees"]:
            avg_best_params = pd.DataFrame(best_params_list).mean().to_dict()
            avg_best_train_mse = np.mean(best_train_mse_list)
            avg_best_val_mse = np.mean(best_val_mse_list)
            if weight_method == "random_forest":
                print('Random Forest')
            elif weight_method == "gradient_boosting":
                print('Gradient Boosting')
            elif weight_method == "extra_trees":
                print('Extra Trees')
            print("Average Best Params:", avg_best_params)
            print("Average Best Train MSE:", avg_best_train_mse)
            print("Average Best Validation MSE:", avg_best_val_mse)
            print('----------')

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
            drawdown = FinancialMetrics.FinancialMetrics.drawdown(returns)["Drawdown"]
            plt.plot(drawdown, label=f"Portfolio Drawdown ({method})")
        index_drawdown = FinancialMetrics.FinancialMetrics.drawdown(index_returns_series)["Drawdown"]
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