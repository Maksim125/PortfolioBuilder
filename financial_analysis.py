import numpy as np
import pandas as pd
from scipy.optimize import minimize
import financial_data


def cov_inv_cov(df):
    """
    Return the covariance and inverse covariance matrices for the assets in the dataframe.
    The variance-covariance matrices are NOT annualized.

    Parameters
    ----------
    df: pd.DataFrame
        Non-cumulative returns of the assets

    Returns
    -------
    (cov, inv_cov) : (pd.DataFrame, pd.DataFrame)
        Covariance and inverse covariance matrices of the returns as a tuple
    """
    cov = df.cov()
    inv_cov = pd.DataFrame(np.linalg.pinv(cov.values), cov.columns, cov.index)
    return cov, inv_cov

def value_at_risk(series, alpha = 0.05):
    sorted_returns = series.sort_values().reset_index(drop=True)
    xth = int(np.floor(alpha * len(sorted_returns))) - 1
    xth_smallest_rate = sorted_returns[xth]
    return xth_smallest_rate


class Allocation:
    def __init__(self, price_df):
        self.noncum_returns = financial_data.noncumul_from_prices(price_df)
        self.cum_returns = financial_data.cumul_from_noncumul(self.noncum_returns)
        if isinstance(self.noncum_returns, pd.Series) or len(self.noncum_returns.columns) == 1:
            self.weights = pd.Series(index = price_df.name, data = [1])
        else:
            self.weights = pd.Series(index = self.noncum_returns.columns, data = [1] + [0]*(len(self.noncum_returns.columns) - 1))
        self.bounds = [(0,1) for _ in range(len(self.weights))]
        self.constraints = {"type" : "eq", "fun" : lambda x : sum(abs(x)) - 1}
    
    #Returns metrics
    def portfolio_raw_return(self, weights):
        """
        Computes the raw returns of the given set of weights
        
        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)

        Returns
        -------
        returns : float
            Percentage returns of these weights for the entire cumulative return period. NOT Annualized. (1% return is expressed as 0.01)
        """
        return (np.sum(weights * self.cum_returns.iloc[-1])) - 1
    
    def portfolio_annual_return(self, weights, annualized = 252):
        """
        Computes the annualized returns of the given set of weights.
        
        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        annualized : int
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)

        Returns
        -------
        returns : float
            Percentage returns of these weights annualized by scaling the returns of this time period. (1% return is expressed as 0.01)
        """
        return (np.sum(weights * self.cum_returns.iloc[-1])**(annualized / len(self.cum_returns.index))) - 1

    def portfolio_annual_alpha(self, weights, risk_free_rate = 0.03, annualized = 252):
        """
        Compute the alpha of this portfolio. (ie. the amount by which annualized returns exceed the annual risk free rate)

        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        risk_free_rate : float
            An annual risk free return. (ie. a 3% risk free annual return is 0.03)
        annualized : int
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)

        Returns
        -------
        alpha : float
            The percent returns exceeding the risk-free rate.
        """
        return self.portfolio_annual_return(weights = weights, annualized = annualized) - risk_free_rate

    #Volatility Metrics
    def portfolio_std_dev(self, weights):
        """
        Computes the standard deviation of the strategy for the given set of weights.
        
        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        
        Returns
        -------
        volatiltiy : float
            The standard deviation of the portfolio
        """
        if isinstance(self.noncum_returns, pd.Series) or len(weights) == 1:
            #There is only 1 asset, return its standard deviation
            return self.noncum_returns.std()
        else:
            #There are multiple assets, return the entire portfolio's standard deviation
            weighted_std = self.noncum_returns.std() * weights
            corr = self.noncum_returns.corr()
            return np.sqrt(np.dot(weighted_std, np.matmul(corr, weighted_std)))

    def portfolio_annual_std_dev(self, weights, annualized = 252):
        """
        Computes the annualized standard deviation of the strategy for the given set of weights.
        
        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        annualized : int
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)
        
        Returns
        -------
        volatiltiy : float
            The annualized standard deviation of the strategy
        """
        std_dev = np.sqrt(annualized) * self.portfolio_std_dev(weights = weights)
        return std_dev

    def portfolio_VAR(self, weights, time_period = 1, alpha = 0.05):
        """
        Compute the value at risk of the given set of weights for a specified tolerance. 

        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        time_period : int
            Period to get value at risk of.
        alpha : float
            Risk tolerance (ie. an alpha of 0.05 will show you the maximum daily drop in 95% of cases) NOT to be confused with returns in excess of the risk free
            rate.

        Returns
        -------
        xth_smallest_rate : float
            The maximum value at risk in (1-alpha)% of cases for returns that have accumulated for the given time period. 
            (ie. if time period is 5, this will be the (1-alpha) percentile cumulative loss in any 5 day period)
        """
        port_noncum_returns = self.portfolio_weight_noncum_returns(weights)
        if time_period > 1:
            #Calculate the cumualtive returns of the time period
            port_noncum_returns = (port_noncum_returns + 1).rolling(int(time_period)).cumprod().dropna() - 1
        sorted_returns =  port_noncum_returns.sort_values().reset_index(drop = True)
        xth = int(np.floor(alpha * len(sorted_returns))) - 1
        xth_smallest_rate = sorted_returns[xth]
        #Return the value at risk, if there exists a period of negative returns
        #Otherwise return 0 since no value is at risk
        value_at_risk = - min(xth_smallest_rate, 0)
        return value_at_risk

    def portfolio_turbulence_series(self,weights, time_period = 1):
        """
        Computes the turbulence of the portfolio per time period.

        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        time_period : int
            Smooth turbulence measures across this time frame

        Returns
        -------
        turbulences : pd.Series
            Series of daily turbulence values
        """
        de_meaned_returns = self.noncum_returns - self.noncum_returns.mean(axis = 0)
        if time_period > 1:
            #Smooth the de-meaned returns if time period is provided.
            de_meaned_returns = de_meaned_returns.rolling(int(time_period)).mean()
        weights_diagonal = pd.DataFrame(np.diag(weights), index = self.noncum_returns.columns,columns=self.noncum_returns.columns)
        coeff = 1 / np.sum(weights**2)
        _, inv_cov = cov_inv_cov(de_meaned_returns)
        inv_cov = inv_cov / (len(de_meaned_returns.index) - 1)
        turbulences = de_meaned_returns.apply(lambda row : np.sqrt(coeff * row.dot(weights_diagonal).dot(inv_cov).dot((row.dot(weights_diagonal)).transpose())), axis = 1)
        return turbulences

    def portfolio_average_turbulence(self, weights, time_period = 1):
        """
        Compute the averge turbulence of the data

        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        time_period : int
            Smooth turbulence measures across this time frame

        Returns
        -------
        average_turbulence : float
            The average turbulence of the portfolio
        """
        turbulence = np.mean(self.portfolio_turbulence_series(weights = weights, time_period = time_period))
        return turbulence
    
    #Optimizers
    def weights_scipy_minimize(self, objective_func):
        """
        Given an objective function that requires weights as its first and only required argument, find the set of weights that will minimize it.

        Parameters
        ----------
        objective_func : a callable function with weights as its first and only required argument that returns a scalar or object that can be compared
            for inequality.

        Returns
        -------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        
        """
        res = minimize(fun = objective_func, x0 = self.weights, method = None, bounds = self.bounds, constraints = self.constraints)
        if not res.success:
            print("SCIPY Failed To Find a Solution")
        return pd.Series(index = self.cum_returns.columns, data = res.x)
    
    def weights_MCMC_minimize(self, objective_func, steps = 1e4, step_size = 0.01):
        """
        A Markov Chain Monte Carlo style solver that will do a random walk toward a solution

        Parameters
        ----------
        objective_func : function
            A function to minimize, it must have weights as its first and only required input

        Returns
        -------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        
        weight_history : list
            A list of weights that the MCMC stepper visited in order
        """
        mcmc = MCMC(dims = len(self.weights),step_size = step_size, objective_func = objective_func)
        mcmc.run(num_steps = steps)
        return pd.Series(data = mcmc.weight_history[-1], index = self.noncum_returns.columns), mcmc.weight_history

    def weights_uniform_sampling_minimize(self, objective_func, samples = 1000, concentration_penalty = False):
        """
        Uniformly sample from the dirichlet distribution and return whichever randomly selected weight minimizes the objective function the best.
        Given it's universal applicability regardless of the mathematical properties of the objective function, this method supports penalties
        for over-concentrating assets

        Parameters
        ----------
        objective_func : function
            A function to minimize, it must have weights as its first and only required input
        samples : int
            The number of samples to draw
        concentration_penalty : bool
            Whether or not to apply a penalty for over-concentrating your portfolio

        Returns
        -------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        """
        sample_weights = np.random.dirichlet(alpha = (1,)*len(self.weights), size = int(samples))

        #Create a reference minimum for the default weight
        minimum = objective_func(self.weights)
        min_weights = None
        filt = lambda x, w : x if not concentration_penalty else x * self.concentration_penalty(w)
        #Go through the samples and find whichever one minimizes the objective function the most
        for weight in sample_weights:
            contendor = filt(objective_func(weight),weight)
            if contendor < minimum:
                min_weights = weight
                minimum = contendor

        return pd.Series(data = min_weights, index = self.weights.index) if min_weights is not None else self.weights

    #Useful attributes and data
    def weighted_cumulative_returns(self, weights):
        """
        Given a set of weights, return a series of the cumulative returns of the portfolio as a whole. Useful for plotting the time
        evolution of the portfolio in the given range.
        
        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        
        Returns
        -------
        cumulative_returns : pd.Series
            Cumulative returns of the given set of weights
        """
        return (self.cum_returns * weights).sum(axis = 1)

    def weighted_noncumulative_returns(self, weights):
        """
        Given a set of weights, return a series of the non-cumulative returns of the portfolio as a whole unit

        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
        
        Returns
        -------
        non_cumulative_returns : pd.Series
            Non-cumulative returns of the given set of weights
        
        """
        return (self.noncum_returns * weights).sum(axis = 1)

    #Common Objective Functions
    def objective_sharpe_ratio(self, annualized = 252, risk_free_rate = 0.03):
        """
        Generates a function compatible with the solvers that optimizes for the sharpe ratio. It will fix all optional
        parameters for the functions necessary to compute the sharpe ratio, while leaving only weights as an independent variable.
        Do not pass this function itself to the solver, pass its output to the solver.

        Parameters
        ----------
        annualized : int
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)
        risk_free_rate : float
            Risk free rate of returns expressed as a decimal. (ie. a 5% risk free rate is 0.05)

        Returns
        -------
        objective_func : function
            An optimizing function compatible with the class's solvers
        """
        #Fix inputs other than weights for the alpha and std dev functions
        objective_func = lambda w : (-1 * self.portfolio_annual_alpha(weights = w, risk_free_rate = risk_free_rate, annualized = annualized) / 
        self.portfolio_annual_std_dev(weights = w, annualized = annualized))
        return objective_func
    
    def objective_std_deviation(self, annualized = 252):
        """
        Generates a function compatible with the solvers that minimizes the standard deviation. It will fix all other
        parameters for the composite functions, while leaving only weights as an independent variable.
        Do not pass this function itself to the solver, pass its output to the solver.

        Parameters
        ----------
        annualized : int
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)

        Returns
        -------
        objective_func : function
            An optimizing function compatible with the class's solvers
        """
        objective_func = lambda w : self.portfolio_annual_std_dev(weights = w, annualized = annualized)
        return objective_func
    
    def objective_alpha_per_VAR(self, annualized = 252, risk_free_rate = 0.03, time_period = 1, alpha = 0.05):
        """
        Generates a function compatible with the solvers that maximizes returns per unit Value At Risk. It will fix all other
        parameters for the composite functions, while leaving only weights as an independent variable.
        Do not pass this function itself to the solver, pass its output to the solver.
        
        Parameters
        ----------
        annualized : int = 252
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)
        risk_free_rate : float
            Risk free rate of returns expressed as a decimal. (ie. a 5% risk free rate is 0.05)
        time_period : int = 1
            Period to get value at risk of.
        alpha : float = 0.05
            Risk tolerance (ie. an alpha of 0.05 will show you the maximum daily drop in 95% of cases) NOT to be confused with returns in excess of the risk free
            rate.
        
        Returns
        -------
        objective_func : function
            An optimizing function compatible with the class's solvers                        
        """
        objective_func = lambda w : (-1 * self.portfolio_annual_alpha(weights = w, risk_free_rate = risk_free_rate, annualized = annualized) / 
        self.portfolio_VAR(weights = w, time_period = time_period, alpha = alpha))
        return objective_func

    def objective_VAR(self, time_period = 1, alpha = 0.05):
        """
        Generates a function compatible with the solvers that minimizes Value At Risk. It will fix all other
        parameters for the composite functions, while leaving only weights as an independent variable.
        Do not pass this function itself to the solver, pass its output to the solver.

        Parameters
        ----------
        annualized : int = 252
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)
        time_period : int = 1
            Period to get value at risk of.
        alpha : float = 0.05
            Risk tolerance (ie. an alpha of 0.05 will show you the maximum daily drop in 95% of cases) NOT to be confused with returns in excess of the risk free
            rate.        

        Returns
        -------
        objective_func : function
            An optimizing function compatible with the class's solvers
        """
        objective_func = lambda w : self.portfolio_VAR(weights = w, time_period = time_period, alpha = alpha)
        return objective_func

    def objective_alpha_per_turbulence(self, annualized = 252, risk_free_rate = 0.03, time_period = 1):
        """
        Generates a function compatible with the solvers that maximizes alpha per unit average turbulence. It will fix all other
        parameters for the composite functions, while leaving only weights as an independent variable.
        Do not pass this function itself to the solver, pass its output to the solver.

        Parameters
        ----------
        annualized : int = 252
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)
        risk_free_rate : float
            Risk free rate of returns expressed as a decimal. (ie. a 5% risk free rate is 0.05)
        time_period : int = 1
            Period to get value at risk of.

        Returns
        -------
        objective_func : function
            An optimizing function compatible with the class's solvers
        """
        objective_func = lambda w : (-1 * self.portfolio_annual_alpha(weights = w, risk_free_rate = risk_free_rate, annualized = annualized) / 
        self.portfolio_average_turbulence(weights = w, time_period = time_period))
        return objective_func

    def objective_mean_turbulence(self, time_period = 1):
        """
        Generates a function compatible with the solvers that minimizes average turbulence. It will fix all other
        parameters for the composite functions, while leaving only weights as an independent variable.
        Do not pass this function itself to the solver, pass its output to the solver.

        Parameters
        ----------
        annualized : int = 252
            The number of time units in the data that exist in a year. (ie. if data is in days, and it is for stocks, 
            this value is 252 because there are 252 trading days in a year in the US stock market)
        risk_free_rate : float
            Risk free rate of returns expressed as a decimal. (ie. a 5% risk free rate is 0.05)
        time_period : int = 1
            Period to get value at risk of.

        Returns
        -------
        objective_func : function
            An optimizing function compatible with the class's solvers
        """
        objective_func = lambda w : self.portfolio_average_turbulence(weights = w, time_period = time_period)
        return objective_func

    #Plotting
    def plot_efficient_frontier(self, ax, return_func, volatility_func, num_points = 1000):
        """
        Plot the efficient frontier by uniformly sampling from the dirichlet distribution

        Parameters
        ----------
        ax : Matplotlib.pyplot axis
        return_func : function
            A function to calculate the returns given a set of weights as input
        volatility_func : function
            A function to calculate volatility given a set of weights as input
        num_points : int
            The number of points to sample and plot
        """
        N = len(self.weights)
        num_points = int(num_points) #In case a float is passed
        sample_weights = np.random.dirichlet(alpha = (1,)*N, size = num_points)
        x_axis = []
        y_axis = []
        for weight in sample_weights:
            y = return_func(weight)
            x = volatility_func(weight)
            y_axis.append(y)
            x_axis.append(x)
        ax.scatter(x_axis, y_axis, label = f"Efficient Frontier {num_points} Samples")

    #Penalties
    def concentration_penalty(self, weights):
        """
        A multiplier that scales the objective function's output depending on how over-concentrated it is.
        The maximum value of the weights is input into the activation function that has effectively 0 penalty up to a 50% allocation, but
        then rapidly climbs the closer you get to 1. No penalty means the penalty will be a 1x multiplier, which will effectively leave the
        weights untouched.

        Parameters
        ----------
        weights : pd.Series
            Allocation of assets expressed as a decimal. Sums to 1 and assumes no shorts. (ie. a 5% allocation is 0.05, and for each weight w, 0 <= w <= 1)
            Index of weights (tickers for each stock) must match columns of returns and index of correlation matrix. (ie. don't drop stocks with 0 weight)
                
        Returns
        -------
        penalty_multiplier : float
            A multiplier that takes the objective function closer to 0 the more over-concentrated the weights are
        """
        func = lambda x : 1 / (1 + 3.1*np.exp(-8.4*x + 6.5))
        penalty_multiplier = 1 - func(max(weights))
        return penalty_multiplier


class Compare_Allocations:
    def __init__(self, price_history, train_fraction = 0.5):
        split = int(len(price_history.index) * train_fraction)
        train_history = price_history.iloc[:split]
        test_history = price_history.iloc[split:]
        assert len(train_history.index) + len(test_history.index) == len(price_history.index), "Split histories don't contain all values"
        self.training_allocation = Allocation(train_history)
        self.testing_allocation = Allocation(test_history)

    def get_train_test_split(self):
        return self.training_allocation, self.testing_allocation
    
    def get_predictive_efficacy(self, ideal_weights, predicted_weights, test_objective):
        """
        Returns the difference between the ideal and predicted objectives over the difference between ideal and
        even-weighted objectives.

        Parameters
        ----------
        ideal_weights : pd.Series
            The weights corresponding to the ideal position for the testing allocation
        predicted_weights : pd.Series
            The weights corresponding to the ideal position for the training allocation, which is what will be
            used for predicting the weight for the testing allocation.
        test_objective : function
            A function whose first and only required input is a set of weights and that yields a scalar value

        Returns
        -------
        percent_distance : float
            The percent distance between the predicted and true objective return. The value is in [-100, 100], and 
            postive values mean the prediction yielded a higher value than ideal, and negative means the prediction yielded lower
            than ideal values
        """
        N = len(self.testing_allocation.weights)
        dummy_weights = np.array([1 / N for _ in range(N)])
        return (abs(test_objective(ideal_weights)) - abs(test_objective(predicted_weights))) / (abs(test_objective(ideal_weights)) - abs(test_objective(dummy_weights)))


class MCMC:
    def __init__(self, dims, objective_func, step_size, burn_in = 100):
        self.dimensions = dims
        self.objective_func = objective_func
        self.step_size = step_size
        self.curr_min = None
        self.weight_history = []
        self.bounds = (0,1)
        self.initial_weights = self.generate_initial_weights(burn_in)

    def generate_initial_weights(self, burn_in):
        """
        Generates a starting point for the walk by drawing uniform samples from a dirichlet
        distribution and picking the one that minimizes the objective function the best. The
        more burn_in samples used, the more likely the walk is to start off close to the ideal location
        but it is costly to perform.

        Parameters
        ----------
        burn_in : int
            How many random samples to draw and go through
        """
        #Generate an amount of random samples from a dirichlet distribution uniformly
        samples = np.random.dirichlet(alpha = (1,)*self.dimensions, size = burn_in)

        #Initialize a dummy weight to serve as a reference minimum
        weight = np.array([1] + [0]*(self.dimensions - 1))
        minimum = self.objective_func(weight)
        self.curr_min = minimum

        #Go through random samples and find the best starting point
        for sample in samples:
            candidate = self.objective_func(sample)
            if candidate < minimum:
                weight = sample
                minimum = candidate
                self.curr_min = minimum
        #Append this starting point as the begining of the walk history
        self.weight_history.append(weight)
    
    def step(self):
        """
        Attempts 1 step in the chain- it will select a random point close to its current location
        on the simplex, and if this point minimizes the objective function more than the current point, 
        it is accepted as the next step. Otherwise nothing is done.
        """
        weight = self.weight_history[-1]
        #Draw a random sample near the last location and rescale them
        #to find a random point "near" the last one
        next = np.random.normal(loc = weight, scale = self.step_size)
        next[next > self.bounds[1]] = self.bounds[1]
        next[next < self.bounds[0]] = self.bounds[0]
        next = next / sum(next)
        
        #If the candidate is better than the current value, it is accepted
        candidate_min = self.objective_func(next)
        if self.curr_min > candidate_min:
            self.curr_min = candidate_min
            self.weight_history.append(next)

    def run(self, num_steps):
        """
        Start the walk by attempting "num_steps" steps.

        Parameters
        ----------
        num_steps : int
            Number of steps to attempt in the walk.
        
        Returns
        -------
        ratio_accepted : float
            The number of steps that were accepted over the amount that were attempted.
        """
        for _ in range(int(num_steps)):
            self.step()

        ratio_accepted = len(self.weight_history) / num_steps
        return ratio_accepted

