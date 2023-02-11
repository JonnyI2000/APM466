#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:58:18 2023

@author: jonnyisenberg
"""

import pandas as pd 
import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import optimize

raw_d = pd.read_csv("bond_data_good.csv")

#data_f.sort_values(by=["Coupon"])
data_f = raw_d[raw_d.columns[:-1]].copy()
data_f.Coupon = raw_d.Coupon.astype(str).apply(lambda row: float(row[:-1]))
data_f["MATURITY_DATE"] = pd.to_datetime(data_f['MATURITY_DATE'])
data_f["ISSUE_DATE"] = pd.to_datetime(data_f['ISSUE_DATE'])
data_f.sort_values(by=["MATURITY_DATE"])



# Selecting bonds and making a new dataframe to store then in.
indices = [0,14,1,20,22,18,28,29,31,35]
ten_bonds = data_f.iloc[indices]

# Copying this new datafram of selected bonds (one for yield and one for 
# pricing).
yield_frame = ten_bonds.copy()
pricing_frame = ten_bonds.copy()

# Setting a variable for the notional value.
notional = 100

class Bonds():
    def pricer(self, coupon, notional, int_rate, years, freq=2):
        total_coupons_pv = self.pv_coups(coupon, int_rate, years, freq)
        npv = self.face_pv(notional, int_rate, years)
        result = total_coupons_pv + npv 
        return result 
        
    def face_pv(self, notional, interest_rate, years):
        fvpv = notional / (1 + interest_rate)**years
        return fvpv
    
    def pv_coups(self, coupon, int_rate, years, freq=2):
        pv = 0 
        for period in range(int(years*freq)):
            pv += self.pv_coup(coupon, int_rate, period +1, freq=2)
        return pv 
    
    def pv_coup(self, coupon, int_rate, period, freq=2):
        pv = (coupon/freq)/(1+int_rate/freq)**period
        return pv 
    
    def ytm(self, bond_price, notional, coupon, years, freq=2, estimate=0.05):
        get_yield = lambda int_rate: self.pricer(coupon, notional, int_rate, years, freq)-bond_price
        return optimize.newton(get_yield, estimate)


# Calculating yield and inputting it to the yield data frame created above
for i in range(yield_frame.shape[0]):
    coup = Bonds()
    for j in yield_frame.columns[-10:]:
        yield_frame[j].iloc[i] = 100*coup.ytm(ten_bonds[j].iloc[i], notional, ten_bonds.Coupon.iloc[i], (i+1)/2, 2)
        

# Calculating the spot rate.
def spot_rater(face, price, coupon, spot_rates, freq):
    value = price
    for i in range(len(spot_rates)):
        value -= (coupon/freq)/((1+(spot_rates[i]/freq))**(i+1))
    last_period = len(spot_rates)
    
    new_rate = -math.log(value/(face+(coupon/freq)))/last_period
    return new_rate


# Setting the notional price
notional = 100 
freq = 2
spot_df = yield_frame.copy()
for column in spot_df.columns[-10:] :
    spot_rates = [(yield_frame[column].iloc[0])/100]
    for i in range(1,spot_df.shape[0]):
        spot_rate = spot_rater(notional, pricing_frame[column].iloc[i], pricing_frame["Coupon"].iloc[i], spot_rates, freq)
        spot_rates += [spot_rate]
        spot_df[column].iloc[i] = 100*spot_rate
        
forward_spots = spot_df.iloc[range(3,10,2)]

spots = forward_spots[forward_spots.columns[-10:]].to_numpy()/100
forwards = np.zeros(spots.shape)

for i in range(4):
    temp = spots + 1
    forwards[i,:] = (np.power(temp[i,:], i+1)/(np.power(temp[0,:],1)))-1
forwards = forwards*100
axis = ['1yr-1yr','1yr-2yr','1yr-3yr','1yr-4yr']
years = ['2025', '2026', '2027', '2028']

yields = yield_frame[yield_frame.columns[-10:]].to_numpy()/100

log_y_returns = np.zeros((9,10))

for i in range(9):
    log_y_returns[i, :] = np.log(yields[:, i+1]) - np.log(yields[:, i])

yield_cov = np.cov(log_y_returns.T)

f = forwards.T
log_fwd_returns = np.zeros(f.shape)
f = np.abs(f)

for i in range(3):
    if i != 0:
        log_fwd_returns[:, i] = np.log(f[:, i+1]) - np.log(f[:, 1])
    else:
        log_fwd_returns[:, i] = np.log(f[:, i+1])

forward_cov = np.cov(log_fwd_returns)
        
yield_eig = np.linalg.eigh(yield_cov)
Forward_eig = np.linalg.eigh(forward_cov)

# ------ PLOTS! --------

# 1. Histogram depicting the distributon of coupon rates across the whole bond
# dataset.
coupons = data_f["Coupon"].values
plt.hist(coupons, bins=30)
plt.xlabel("Coupon Value")
plt.ylabel("Frequency")
plt.title("Histogram of Coupon Values")
plt.savefig("Coupon_Hist.pdf")
plt.show()
# 2. Histogram depicting the distributon of maturities across the whole bond
# dataset.
maturities = data_f["MATURITY_DATE"].values
plt.hist(maturities, bins=30)
plt.xlabel("Maturity")
plt.ylabel("Frequency")
plt.title("Histogram of Maturity Dates")
plt.savefig("Maturity_Hist.pdf")
plt.show()
# 3. 5-year yield curve of 10 selected bonds.
for date in yield_frame.columns[-10:]:  
    plt.plot(yield_frame.MATURITY_DATE, yield_frame[date], label = date)
plt.xticks(fontsize=8)
plt.xlabel("Maturity")
plt.ylabel("YTM (%)")
plt.title("5-Year Yield Curve")
plt.savefig("5-Year Yield Curve.pdf")
plt.legend()
plt.show()
# 4. 5-year spot rate curve.
for date in yield_frame.columns[-10:]:  
    plt.plot(spot_df.MATURITY_DATE, spot_df[date], label = date)
plt.xticks(fontsize=8)
plt.xlabel("Maturity")
plt.ylabel("Spot Rate (%)")
plt.title("5-Year Spot Rate Curve")
plt.savefig("5-Year Spot Rate Curve.pdf")
plt.legend()
plt.show()
# 5. 2-5 year forward rate curve.
for i in range(10):
    plt.plot(years, forwards[:,i],label = yield_frame.columns[-i-1])
plt.xlabel("Year of Rate")
plt.ylabel("Forward Rate (%)")
plt.title("2-5 Year Forward Rate Curve")
plt.savefig("2-5 Year Forward Rate Curve.pdf")
plt.legend(loc=4)
plt.show()  


