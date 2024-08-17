# Statistical-Factor-Model

## Project Overview
This project explores the application of Asymptotic Principal Component Analysis (APCA) in creating a statistical factor model aimed at enhancing investment strategies in the Italian stock market, specifically targeting outperformance against the FTSEMIB Index. The initiative bridges sophisticated statistical methodologies with actionable investment tactics to improve portfolio management through deeper insights into asset behavior.

### Core Approach
APCA is employed to address the challenges posed by high-dimensional datasets commonly found in financial markets, where the number of assets often surpasses the number of observable time periods. This analysis technique adapts the traditional Principal Component Analysis (PCA) framework to focus on time periods rather than assets. This shift enables more stable and accurate estimation of the covariance matrix, which is critical for identifying the key factors that influence asset returns.

### Portfolio Strategy Development
APCA is employed to address the challenges posed by high-dimensional datasets commonly found in financial markets, where the number of assets often surpasses the number of observable time periods. This analysis technique adapts the traditional Principal Component Analysis (PCA) framework to focus on time periods rather than assets. This shift enables more stable and accurate estimation of the covariance matrix, which is critical for identifying the key factors that influence asset returns.

### Robustness Testing via Rolling Window Analysis
To ensure the reliability and adaptability of the APCA model, a rolling window testing approach is utilized. This method allows for continuous reassessment and refinement of the model across varying historical data segments, mimicking a dynamic trading environment. Such rigorous testing is essential for confirming the model’s effectiveness over time and across different market conditions.

### Performance Metrics and Evaluations
The success of the investment strategy is quantitatively assessed using a variety of performance and risk metrics, including maximum drawdown, Sharpe ratio, and semi-deviation. These metrics are crucial for evaluating the risk-adjusted returns of the portfolio and for benchmarking its performance against the FTSEMIB Index. The comprehensive analysis helps in identifying the strengths and potential areas for improvement in the investment strategy.

### Conclusion
In evaluating various weighting methods—equal, risk parity, momentum, MaxiMin, stochastic dominance, and gradient boosting—for portfolio factor weighting, all strategies demonstrated outperformance relative to the benchmark in terms of Sharpe ratio. The insights derived from this analysis highlight diverse strategic outcomes depending on the chosen risk and return profiles:

- For investors prioritizing **low Value-at-Risk**, the **equal** and **risk parity** methods prove most effective. These strategies offer robust performance with minimal exposure to extreme losses, as indicated by their lower historic and Cornish-Fisher Value-at-Risk measures.

- **Gradient boosting** stands out for those seeking to minimize **max drawdown**, reducing it dramatically to just 6.8%, coupled with a solid Sharpe ratio of 0.755. This approach is particularly suitable for investors who aim for higher returns but with controlled downside risk.

- **Stochastic dominance** provides a balanced approach with moderate performance across multiple metrics, positioning it as a versatile middle-ground strategy.

- The **MaxiMin strategy** emerges as the standout in terms of both performance and risk metrics, particularly noteworthy for its high net profit to worst drawdown ratio and a favorable information ratio of approximately 0.88. This strategy is likely the most appealing for performance-driven investors with a moderate tolerance for risk.

Incorporating the **Kelly criterion** method into the analysis, the performance markedly improves, but at the cost of a significant increase in max drawdown, reaching up to 48%. This method suits investors with a high-risk appetite, ready to tolerate substantial drawdowns in pursuit of exceptional returns.

This comprehensive evaluation provides a clear decision framework for investors, enabling them to choose strategies that best align with their risk tolerance and investment objectives. Each method's strengths and weaknesses are articulated through detailed risk-return profiling, illustrating the trade-offs involved in optimizing portfolio performance in the Italian stock market.

## Repository Structure

- **models**: this folder includes different Python files with classes about financial metrics, data loading, portfolio weights, and APCA structure.

- **RollingStrategy.py**: this Python script contains the necessary class for the rolling APCA strategy. It includes methods for performing APCA, and evaluating portfolios.

- **Stat_factor_model.ipynb**: open this Jupyter notebook to run the analysis. It provides a comprehensive visual interface for assessing performance over time, viewing drawdowns, and examining summary statistics.

- **Tickers.json**: this file contains all the ticker symbols used in the analysis. You can modify this file to include different tickers, such as switching from Italian to UK tickers, to tailor the analysis to different markets or sectors.

## How to Use
1. Ensure you have Python and pip installed.
2. Install all dependencies listed in `requirements.txt`.
3. Adjust parameters or ticker selections as needed by editing `Tickers.json`.
4. Open the `Stat_factor_model.ipynb` notebook in Jupyter.
5. Run each cell sequentially to load the data, execute the APCA analysis, and generate the visualizations.

## Requirements
To install the required Python libraries, run the following command:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the Apache License 2.0


---

This README provides an overview of the Statistical-Factor-Model repository, including its features, requirements, usage, performance metrics and detailed descriptions of different weighting methods utilized for the different factors.
