# README
# Portfolio Builder
---
A framework for building an optimal financial portfolio full of your favorite stocks, bonds, and cryptocurrencies based on historical patterns. (Checkout a demo in the Outline.ipynb file, and a high-level overview [here](https://www.maxyarmak.tech/portfoliobuilder))
## Relevant Modules

```sh
financial_data.py
financial_analysis.py
```
These two modules contain the methods/classes necessary to use this library.
## Version/Dependencies
```sh
Python 3.8.11
Numpy 1.19.5
Pandas 1.3.1
Yfinance 0.1.63
Matplotlib 3.4.2
Scipy 1.7.1
```
## Feature Summary
- Build portfolio using stock tickers you've selected yourself
- Choose a historical period on which to optimize a specific metric (returns, volatility, returns per volatility, etc)
- Choose an optimizer to find the ideal distribution for your chosen metric
- Optimizer returns the stock distribution that has given you the best performance for your chosen metric in the past
- A comparator can show you how well past returns have predicted future returns for your chosen stocks for a given training fraction

## Feature Outline
(Much greater detail and a working demo can be found in the Outline.ipynb file)
- `financial_data.py` contains methods to acquire relevant historical data
    - `prices_from_tickers` acts like a wrapper for the Yfinance download operation, allowing you to access historical returns for your selection of tickers for your selected time frame and makes sure to clean up the data so it has only valid entries
    - `cumul_noncumul_from_tickers` allows you to get normalized cumulative and non-cumulative historical returns for your selection of tickers for your selected time frame
    - Other methods let you convert between raw prices, cumulative and non-cumulative returns as you need them
- `financial_analysis.py` contains methods to analyze data formatted by the aforementioned module
    -   `Allocation` class is constructed by inputting the direct result from `prices_from_tickers`
        -   Once an object (ex. `my_allocation = Allocation(price_history)`) has been created using historical returns you can use its methods to get useful data: (for each of the below functions you need to create your own set of portfolio weights)
            -   `my_allocation.portfolio_raw_return(weights)` shows you the returns of the portfolio for the given time frame
            -   `my_allocation.portfolio_annual_return(weights)` shows you annualized returns
            -   `my_allocation.portfolio_annual_alpha(weights)` shows you annualized returns in excess of the risk free rate
            -   `my_allocation.portfolio_annual_std_dev(weights)` shows you an annualized standard deviation
            -   `my_allocation.portfolio_VAR(weights)` shows you the value at risk measure
            -   `my_allocation.portfolio_turbulence_series(weights)` will return the daily turbulence measure in a series
            -   `my_allocation.portfolio_average_turbulence(weights)` will return the average turbulence for the time period
        - Solve for optimal weights using one of the `Allocation`'s solvers
            - Generate an objective to minimize by either calling on one of the pre-built objective builders or constructing your own.
                - Prebuilt Objectives Include:
                - `objective_sharpe_ratio` Maximizes the sharpe ratio
                - `objective_std_deviation` Minimizes the standard deviation
                - `objective_VAR` minimizes the value at risk
                - `objective_alpha_per_VAR` maximizes the returns in excess of the risk free rate per unit value at risk
                - `objective_alpha_per_turbulence` maximizes the returns in excess of the risk free rate per unit mean turbulence
            - Prebuilt objectives can be called with no arguments, or with some or all of the optional arguments that let you modify their behavior
            - Custom objectives can be made as well: as long as you pass in a method that takes in weights as the first and only required parameter and returns a scalar or comparable object of some kind, the solvers will minimize this value. (maximizing can be done by just negating the value you want to maximize)
        - Solvers:
            -  `scipy.optimize.minimize` has a wrapper in the `Allocation` class called `weights_scipy_minimize` that lets you pass in a valid objective function and minimize its value
            -  `weights_MCMC_minimize` is a custom built solver that uses the Markov Chain Monte Carlo method to reach a solution. "burn-in" is accomplished by uniform sampling of a few weights from the dirichlet distribution and picking the best starting point from there. Steps are taken by sampling weight deviations from a standard normal distribution and walking toward the ideal solution.
            -  `weights_uniform_sampling_minimize` is the brute-force method of the bunch that simply uniformly samples a very large number of points from the dirichlet distribution and picks whichever one optimizes for the objective the best. This method should only be used as an approximate sanity check with reasonably few samples, as getting to the ideal solution with this method is extremely inefficient.
    -  `Compare_Allocations` class is built by inputting a price history and a fraction that represents the proportion of data that will be used for the train split. 
        -  `get_predictive_efficacy` returns a value that serves as a proxy for how well optimizing on train data will predict performance for future data. (Ideally this value should be as close to `0` as possible, where any value above `1` means a random model outperforms your optimizations.)
