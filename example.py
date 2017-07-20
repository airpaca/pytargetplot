#!/usr/bin/env python3.6
# coding: utf-8

"""Example of use of the `pytargetplot` library."""


import pandas
from pytargetplot import ModelPerformance


# Read measure and modeling data
df = pandas.read_csv('data/example.csv')

# Compte model performance
# It will compute measure uncertainty if not available
mp = ModelPerformance(df, param='NO2',
                      field_obs='observation', field_mod='model')

# Statistical information
print('n =', mp.n)
print('bias =', mp.bias)
print('r =', mp.r)

# MQO : Model Quality Objective
print('MQO =', mp.mqo)

# Summary of all statistical informations
print(mp.summary)

# Contingency table
print(mp.contingency_table(limit=50))

# Create a Target Plot
mp.create_target_plot()

