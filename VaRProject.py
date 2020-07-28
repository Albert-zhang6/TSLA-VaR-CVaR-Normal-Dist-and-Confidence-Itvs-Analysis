import quandl
import pandas as pd
import numpy  as np
import math
import matplotlib.pyplot as plt
import datetime as dt


start = dt.datetime(2010, 7, 29)
end = dt.datetime.now()

df = pd.read_csv("C:/Users/Albert Zhang/Desktop/test/TSLA.csv")


#Computing Daily Return

df['Daily Return'] = np.log(df['Close']/df['Close'].shift(1))


#making normal distribution

from scipy.stats import norm

mean = df['Daily Return'].mean()
std = df['Daily Return'].std()
counter = df['Daily Return'].count()

df2 = pd.DataFrame()

df2['x'] = np.arange(-0.05, 0.05, 0.001)
df2['norm'] = norm.pdf(df2['x'],mean,std)

plt.plot(df2['x'],df2['norm'], color='b')
plt.show()

#Calculating Value at Risk - #5% chance that the daily return will be less than ____
print()

Var_5 = norm.ppf(0.05, mean, std)
print("5% VaR = " + str(Var_5))

print()

proba_5 = norm.cdf(-0.05, mean, std) #chance that the daily return will be less than 5% 
print("chance that the daily return will be less than 5%: " + str(proba_5))

print()

#Calculating the confidence interval at 95.5%

low=mean-2*std/np.sqrt(counter)
up=mean+2*std/np.sqrt(counter)

print("There is a 95.5% probability that daily returns will fall between " + str(low) +  " and " + str(up))


#Calculating CVaR



df.loc[df['Daily Return'] < 0, 'Losses'] = df['Daily Return']

nLoss = df['Losses'].count()

sumLoss = df['Losses'].sum()

meanLoss = df['Losses'].mean()
stdLoss = df['Losses'].std()

# Compute the expected tail loss and the CVaR in the worst 5% of cases
tail_loss = norm.expect(lambda x: x, loc = meanLoss, scale = stdLoss, lb = Var_5)
CVaR_95 = (1 / (1 - 0.95)) * tail_loss

print()

print("The expected return on TSLA for the worst 5% of cases (CVAR) = " + str(CVaR_95))

# Plotting the normal distribution histogram and adding lines for the CVaR and VaR
plt.hist(norm.rvs(size = 100000, loc = meanLoss, scale = stdLoss), bins = 100)
plt.axvline(x = Var_5, c='r', label = "VaR, 95% confidence level")
plt.legend(); plt.show()