import numpy as np
import pandas as pd
from scipy.stats import norm

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
