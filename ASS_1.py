#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:58:18 2023

@author: jonnyisenberg
"""

from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Importing bond names
bondnames = np.loadtxt('./bond_data.csv', dtype=str, delimiter=",", skiprows=1, usecols=0, unpack=True)

# Cleaning up bond names
count = 0
for i in bondnames:
    new = i[1:]
    bondnames[count] = new
    count+=1
    
# importing bond prices
bondprices = np.round(np.array(np.loadtxt('./bond_data.csv', dtype=float, \
                                          delimiter=",", skiprows=1, \
                                              usecols=range(6,14), \
                                                  unpack=True)), 2)
# Importing maturity dates
mats = np.loadtxt('./bond_data.csv', dtype='str', delimiter=",", skiprows=1, usecols=2, unpack=True)
# Importing coupon rates
coups = np.loadtxt('./bond_data.csv', dtype='float', delimiter=",", skiprows=1, usecols=1, unpack=True)
    

# Storing names and prices together
bond_prices = {}
for i in bondnames:
    name = i
    lst=[]
    for j in bondprices:
        lst.append(j[0])
    bond_prices[name] = lst
    
# How long until maturity?
for i,j,h in zip(mats, bond_prices, coups):
    name = j
    dates = []
    count=16
    for k in bond_prices[name]:
        dates.append(((date(int(i[0:4]), int(i[4:5]), int(i[5:]))-date(2023,1,count)).days/365))
        count+=1
    bond_prices[name] = (bond_prices[name], dates, h)
    

# Calculating yield at close for each bond on each day
x_initial_guess = 0.01

bond_yields = {}

for i in bond_prices:
    name = i
    # List of yields
    lst = []
    # List of time-deltas
    lst_2 = []
    for j,k in zip(bond_prices[name][0], bond_prices[name][1]):
        if k < 0.5:
            yield_ = (-np.log(j/100)/(k))
            lst.append(yield_)
            lst_2.append(k) 
        else:
            coupons = round(k*2, 0)
            def equation(x):
                return (bond_prices[name][2]/x)*(1-(1+x)**(-coupons))+(100/(1+x)**coupons) - j
            x_initial_guess = 0.01
            x_solution = fsolve(equation, x_initial_guess)
            lst.append(x_solution)
            lst_2.append(k)
    bond_yields[name] = (lst, lst_2)
    
price = (1.5/0.01549)*(1-(1+0.01549)**(-6))+(100/(1+0.01549)**6)

# =============================================================================
# (coupon / (ytm/2)) * (1 - (1 + ytm/2)^(-2 * maturity)) + face value / (1 + ytm/2)^(2 * maturity)
# =============================================================================
    
# =============================================================================
# accru = 100*(coupon/2)*(date)
# =============================================================================
    
# Calculating the slope of yields for each bond 


    
fig, ax = plt.subplots()
for i in bond_yields:
    name = i
    ax.plot(bond_yields[name][1], bond_yields[name][0], label=name)

ax.set_xlabel("Time (in years)")
ax.set_ylabel("Yield (%)")
ax.legend()

plt.show()


# =============================================================================
# x_1 = np.array([bond_yields[i][1] for i in bond_yields])
# y_1 = np.array([bond_yields[i][0] for i in bond_yields])
# 
# plt.scatter(x_1[4],y_1[4])
# =============================================================================


    

    
        


# =============================================================================
# for i in range(len(bondnames[0])):
#     bond_prices[bondnames[0][i]] = (bondprices[i])
# 
# print(bond_prices)
# =============================================================================



#CAN_1_5_JUN_1_23 = {"CAN 1.5 JUN 1 23": ("1.5", date(2023, 6, 1))}
