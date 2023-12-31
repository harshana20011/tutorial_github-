import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end)['Close']
    returns = stock_data.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix


# Liste de stocks américains
stock_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM']
stocks = stock_list  # Pas besoin d'extension spécifique pour les bourses américaines
# Apple (AAPL), Microsoft (MSFT), Alphabet (GOOGL), Amazon (AMZN), Tesla (TSLA), et JPMorgan Chase (JPM)


end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=364 * 2)

try:
    mean_returns, cov_matrix = get_data(stocks, start_date, end_date)
    # print("Mean Returns:")
    # print(mean_returns)
    # print("\nCovariance Matrix:")
    # print(cov_matrix)
    weights = np.random.random(len(mean_returns))
    # normalisation :
    weights /= np.sum(weights)
    print(weights)
    initial_portfolio = 10000
    # Monte Carlo
    mc_sims = 100
    T = 364 * 2
    mean_m = np.full(shape=(T, len(weights)), fill_value=mean_returns)
    mean_m = mean_m.T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    for m in range(0, mc_sims):
        # we assume that daily returns are distributed by a multivariate normal distribution
        # Cholesky Decomposition lower triangle
        # uncorralated sample data and we corrolate it with cov matrix and lower triangle
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(cov_matrix)
        daily_returns = mean_m + np.inner(L, Z)
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, daily_returns.T)+1)*initial_portfolio
    plt.plot(portfolio_sims)
    plt.ylabel('portfolio value')
    plt.xlabel('Days')
    plt.title('MC simulation of a portfolio (stock)')
    plt.show()


except ValueError as e:
    print(e)
