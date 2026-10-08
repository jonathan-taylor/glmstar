# ---
# jupyter:
#   jupytext:
#     formats: ipynb,Rmd,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
from glmnet.cox import CoxLM, CoxNetIRLS, CoxFamilySpec
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statsmodels.api as sm
import rpy2
# %load_ext rpy2.ipython

rng = np.random.default_rng(0)
from glmnet.glm import GLMControl
from glmnet.glmnet import GLMNetControl
control = GLMControl(epsnr=1e-8)
GNcontrol = GLMNetControl()
# -

GNcontrol

n, p = 831, 20
status = rng.choice([0, 1], size=n)
start = rng.integers(0, 5, size=n)
event = start + rng.integers(1, 5, size=n) 
event_data = pd.DataFrame({'event':event,
                           'status':status,
                           'start':start})
X = rng.standard_normal((n, p))
event_data
W = rng.integers(2, 6, size=n) + rng.uniform(0, 1, size=n)
W[:20] = 3
event_data['weight'] = W

# # Check a GLM

breslow = CoxFamilySpec(event_data,
                        event_id='event',
                        status_id='status',
                        start_id='start',
                        tie_breaking='breslow')
efron = CoxFamilySpec(event_data,
                      event_id='event',
                      status_id='status',
                      start_id='start',
                      tie_breaking='efron')
G1 = CoxLM(family=breslow,
           weight_id='weight')
G1.fit(X, event_data)

# ## Also using `pd.DataFrame`

G2 = CoxLM(weight_id='weight', family=breslow, summarize=True)
G2.fit(X, event_data)
assert np.allclose(G2.coef_, G1.coef_)
assert np.allclose(G2.intercept_, G1.intercept_)
G2.summary_
#
#

# + magic_args="-i event,status,start,X,W -o C" language="R"
# library(survival)
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.numeric(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# M = coxph(Y ~ X,
#           weight=W,
#           ties='breslow',
#           robust=FALSE)
# print(summary(M))
# C = coef(M)
# -

# ## Efron's tie breaking

G3 = CoxLM(weight_id='weight', family=efron, summarize=True)
G3.fit(X, event_data)
G3.summary_

# + magic_args="-i event,status,start,X,W -o C" language="R"
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.numeric(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# M = coxph(Y ~ X,
#           weight=W,
#           robust=FALSE)
# print(summary(M))
# C = coef(M)
# -

assert np.allclose(G3.coef_, C, rtol=1e-4, atol=1e-4)


G3.coef_, C

# # CoxNetIRLS
#
# ## Without CV

GN = CoxNetIRLS(family=breslow,
            weight_id='weight',
             ).fit(X, event_data)
GN.summary_

# + magic_args="-o C,L" language="R"
# library(glmnet)
# GN = glmnet(X, 
#             Y, 
#             weights=W, 
#             family='cox')
# C = as.matrix(coef(GN))
# L = GN$lambda
# print(GN)
# -

L.shape, GN.lambda_values_.shape

GN.lambda_max_, L.max(), L.min()

assert np.allclose(C.T[10], GN.coefs_[10], rtol=1e-4, atol=1e-4)

C.T[10]

GN.coefs_[10]

assert np.allclose(C.T[:20], GN.coefs_[:20], rtol=1e-4, atol=1e-4)


assert np.allclose(L[:30], GN.lambda_values_[:30])

# ## Now with CV, first no weights

GN3 = CoxNetIRLS(family=breslow,
             weight_id='weight',
             control=GNcontrol
             ).fit(X, event_data)


# Capture the fold ids

cv = KFold(5, random_state=0, shuffle=True)
foldid = np.empty(n)
for i, (train, test) in enumerate(cv.split(np.arange(n))):
    foldid[test] = i+1

# ## Using `fraction`

predictions, scores = GN3.cross_validation_path(X[:,:8], event_data, cv=cv, alignment='fraction');

GN3.plot_cross_validation(score='Cox Deviance (Difference)', xvar='lambda')

# + magic_args="-i foldid,X,event,start,status,W -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.integer(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# GCV = cv.glmnet(x=X[,1:8],
#                 y=Y,
#                 weights=W,
#                 family='cox',
#                 foldid=foldid,
#                 alignment='fraction',
#                 grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Cox Deviance (Difference)'].iloc[:10], CVM[:10], rtol=1e-3, atol=1e-3)
assert np.allclose(GN3.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10], CVSD[:10], rtol=1e-3, atol=1e-3) 

fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['Cox Deviance (Difference)'].iloc[:10], CVM[:10])
ax.axline((GN3.cv_scores_['Cox Deviance (Difference)'].iloc[:10].min(),CVM[:10].min()), slope=1, ls='--', c='k')


fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10], CVSD[:10])
ax.axline((GN3.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10].min(),CVSD[:10].min()),slope=1, ls='--', c='k')


# ## Using `lambda`

# +
GN4 = CoxNetIRLS(family=breslow,
             weight_id='weight',
             control=GNcontrol
             ).fit(X, event_data)

predictions, scores = GN4.cross_validation_path(X[:,:8], event_data, cv=cv, alignment='lambda');
GN4.plot_cross_validation(score='Cox Deviance (Difference)', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.integer(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# GCV = cv.glmnet(x=X[,1:8],
#                 y=Y,
#                 weights=W,
#                 family='cox',
#                 foldid=foldid,
#                 alignment='lambda',
#                 grouped=TRUE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
#
# -

assert np.allclose(GN4.cv_scores_['Cox Deviance (Difference)'].iloc[:10], CVM[:10], rtol=1e-3, atol=1e-3)
assert np.allclose(GN4.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10], CVSD[:10], rtol=1e-3, atol=1e-3) 

fig, ax = plt.subplots()
ax.scatter(GN4.cv_scores_['Cox Deviance (Difference)'].iloc[:10], CVM[:10])
ax.axline((GN4.cv_scores_['Cox Deviance (Difference)'].iloc[:10].min(),CVM[:10].min()),slope=1, ls='--', c='k')


fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10], CVSD[:10])
ax.axline((GN3.cv_scores_['SD(Cox Deviance (Difference))'].iloc[:10].min(),CVSD[:10].min()),slope=1, ls='--', c='k')


# ## Using `fraction`

predictions, scores = GN3.cross_validation_path(X[:,:8], event_data, cv=cv, alignment='fraction');

GN3.plot_cross_validation(score='Cox Deviance', xvar='lambda')

# + magic_args="-i foldid,X,event,start,status,W -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.integer(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# GCV = cv.glmnet(x=X[,1:8],
#                 y=Y,
#                 weights=W,
#                 family='cox',
#                 foldid=foldid,
#                 alignment='fraction',
#                 grouped=FALSE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
# -

assert np.allclose(GN3.cv_scores_['Cox Deviance'].iloc[:10], CVM[:10], rtol=1e-3, atol=1e-3)
assert np.allclose(GN3.cv_scores_['SD(Cox Deviance)'].iloc[:10], CVSD[:10], rtol=1e-3, atol=1e-3) 

fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['Cox Deviance'].iloc[:10], CVM[:10])
ax.axline((GN3.cv_scores_['Cox Deviance'].iloc[:10].min(),CVM[:10].min()), slope=1, ls='--', c='k')


fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['SD(Cox Deviance)'].iloc[:10], CVSD[:10])
ax.axline((GN3.cv_scores_['SD(Cox Deviance)'].iloc[:10].min(),CVSD[:10].min()),slope=1, ls='--', c='k')


# ## Using `lambda`

# +
GN4 = CoxNetIRLS(family=breslow,
             weight_id='weight',
             control=GNcontrol
             ).fit(X, event_data)

predictions, scores = GN4.cross_validation_path(X[:,:8], event_data, cv=cv, alignment='lambda');
GN4.plot_cross_validation(score='Cox Deviance', xvar='lambda')

# + magic_args="-i foldid -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.integer(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# GCV = cv.glmnet(x=X[,1:8],
#                 y=Y,
#                 weights=W,
#                 family='cox',
#                 foldid=foldid,
#                 alignment='lambda',
#                 grouped=FALSE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd
#
# -

assert np.allclose(GN4.cv_scores_['Cox Deviance'].iloc[:10], CVM[:10], rtol=1e-3, atol=1e-3)
assert np.allclose(GN4.cv_scores_['SD(Cox Deviance)'].iloc[:10], CVSD[:10], rtol=1e-3, atol=1e-3) 

fig, ax = plt.subplots()
ax.scatter(GN4.cv_scores_['Cox Deviance'].iloc[:10], CVM[:10])
ax.axline((GN4.cv_scores_['Cox Deviance'].iloc[:10].min(),CVM[:10].min()),slope=1, ls='--', c='k')


fig, ax = plt.subplots()
ax.scatter(GN3.cv_scores_['SD(Cox Deviance)'].iloc[:10], CVSD[:10])
ax.axline((GN3.cv_scores_['SD(Cox Deviance)'].iloc[:10].min(),CVSD[:10].min()),slope=1, ls='--', c='k')


# # When we scale weights, scores stay the same...

# + magic_args="-i foldid,X,event,start,status,W -o CVM,CVSD" language="R"
# foldid = as.integer(foldid)
# event = as.numeric(event)
# start = as.numeric(start)
# status = as.integer(status)
# Y = Surv(start, event, status)
# W = as.numeric(W)
# GCV = cv.glmnet(x=X[,1:8],
#                 y=Y,
#                 weights=W*2,
#                 family='cox',
#                 foldid=foldid,
#                 alignment='fraction',
#                 grouped=FALSE)
# plot(GCV)
# CVM = GCV$cvm
# CVSD = GCV$cvsd

# +
from copy import deepcopy
event_data2 = deepcopy(event_data)
event_data2['weight'] *= 2
GN5 = CoxNetIRLS(family=breslow,
             weight_id='weight',
             control=GNcontrol
             ).fit(X, event_data2)

predictions, scores = GN5.cross_validation_path(X[:,:8], event_data, cv=cv, alignment='lambda');
GN5.plot_cross_validation(score='Cox Deviance', xvar='lambda')
# -


