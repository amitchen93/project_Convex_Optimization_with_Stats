#%%


# %%

from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sb
import numpy as np
import pickle as pkl

from IPython.core.display import display

sb.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (11, 7)
import pandas as pd
import pickle
from numpy import linalg as LA


def dis(L):
    display(pd.DataFrame(L))


class Example:
    def __init__(self, S: np.ndarray, K: np.ndarray, k: int):
        self.k = k
        self.S = S
        self.K = K


with open('examples_01.pkl', 'rb') as f:
    examples = pickle.load(f)

with open('examples_02.pkl', 'rb') as f:
    examples1 = pickle.load(f)

# %% md

## Things to do with regard to caculation optimization:


def calculate_kband_mask(k, n):
    mask = np.zeros((n, n))
    for i in range(n):
        top = i + k
        bottom = i - k
        if i - k <= 0:
            bottom = 0
        if i + k + 1 > n:
            top = n
        mask[i, bottom:top + 1] = 1
    return mask


def calculate_kband_mult(S, L, k, n):
    final_mat = np.zeros((n, n))
    for i in range(n):
        final_mat[:, i] = (S[:, i:i + k + 1] * L[i:i + k + 1, i]).sum(axis=1)
    return final_mat


import pickle


# %%

STABILITY_EPS = 1e-20


def avoid_nurmerical_break(diag_L):
    geq_pos_eps = np.greater_equal(STABILITY_EPS, diag_L)
    geq_pos_zero = np.greater_equal(diag_L, 0)
    indicator_pos = np.logical_and(geq_pos_eps, geq_pos_zero)

    geq_neg2 = np.greater_equal(0, diag_L)
    geq_neg = np.greater_equal(diag_L, -STABILITY_EPS)

    indicator_neg = np.logical_and(geq_neg, geq_neg2)

    diag_L[indicator_pos] = STABILITY_EPS
    diag_L[indicator_neg] = -STABILITY_EPS

    return diag_L


# %%

# Create random data
def generateRandomData(k, n):
    # PSD
    L = np.zeros((n, n))
    L[np.diag_indices(n)] = np.random.rand(n)
    mid_L = np.tril(calculate_kband_mask(k, n)) * np.random.randn(n, n)
    mid_L[np.diag_indices_from(mid_L)] = 0
    L = L + mid_L
    return L


def generateRandomDataS(n):
    # PSD
    L = np.zeros((n, n))
    L[np.diag_indices(n)] = np.random.rand(n)
    mid_L = np.tril(np.ones((n,n))) * np.random.randn(n, n)
    mid_L[np.diag_indices_from(mid_L)] = 0
    L = L + mid_L
    L=L@L.T
    return L


# %%

def calc_g_step(S, L, k, n, mask):
    diag_L = (np.diag(L))  # L_i,i
    inv_diag_L = 2 / diag_L
    avoid_nurmerical_break(inv_diag_L)
    T_1 = calculate_kband_mult(S, L, k, n)
    gradient = (2 * T_1) - np.identity(n) * inv_diag_L
    gradient = gradient * mask
    unorm_grad=gradient
    gradient = gradient / LA.norm(gradient)
    return gradient ,unorm_grad


def GD_total(S, k, searching_steps=5,opt_K_flag=False,opt_K=[],return_grad=False,iter_total=10000,iter_per_lr=100,learn_K=False):

    n = S.shape[0]
    parameters = (2 * k + 1) * n
    itrs = round(300 / np.sqrt(parameters))
    if itrs < 1:
        itrs = 5

    max_range=0.5
    min_range=0.1

    L ,grad_size= init_random_L(0.1, S, k, itrs, 20)
    total_ind=np.zeros(searching_steps)
    y_vals=[]
    err_list=[]

    data_save=np.zeros((iter_total,4))
    for i in range(iter_total):

        L, lr_opt ,opt_ind,lrs_list= search_optimal_lr(L, S, k=k, search_s=searching_steps,range_max=max_range,min_range=min_range)
        L, f ,err,unormed_grad_list= GD_ls(L, S, lr=lr_opt, k=k, iters=iter_per_lr)

        total_ind+=opt_ind
        err_list.append(err)
        y_vals.append(f)
        unormed_grad_list=np.array(unormed_grad_list)


        hat_K = L @ L.T
        if opt_K_flag==True:
            err = np.linalg.norm(opt_K - hat_K, "fro") / (n * k)
            if i%99==0:
                print("err {} , i {}".format(err,i))

            if err<1e-3 and return_grad:
                print("before return i {} and error".format(i,err))
                return unormed_grad_list[-1]

        if learn_K:
            if i%99==0:
                print(unormed_grad_list[-1])
            if unormed_grad_list[-1]<1e-5:
                return hat_K

        data_save[i]=err,unormed_grad_list.mean(),k,n


        if i%1==0:
            max_range=lrs_list[total_ind.argmax()]*1.5
            min_range= lrs_list[total_ind.argmax()]*0.5
            total_ind=np.zeros(searching_steps)


    K=L@L.T

    return K,data_save




def GD_solve(S, k, searching_steps=5,iter_total=10000):


    n=S.shape[0]
    parameters_amount=(2*k+1)*n
    stopping_cond=calc_stopping_crit(parameters_amount)
    iter_per_lr=round(calc_iter_step(parameters_amount))

    if iter_per_lr<20:
        iter_per_lr=20

    init_iter_steps = round(300 / np.sqrt(parameters_amount))
    if init_iter_steps < 1:
        itrs = 5

    L ,grad_size= init_random_L(0.1, S, k, init_iter_steps, 20)
    total_ind=np.zeros(searching_steps)


    max_range=0.5
    min_range=0.1
    for i in range(iter_total):

        L, lr_opt ,opt_ind,lrs_list= search_optimal_lr(L, S, k=k, search_s=searching_steps,range_max=max_range,min_range=min_range)
        L, f ,err,unormed_grad_list= GD_ls(L, S, lr=lr_opt, k=k, iters=iter_per_lr)
        total_ind+=opt_ind
        if i%1==0:
            max_range=lrs_list[total_ind.argmax()]*1.5
            min_range= lrs_list[total_ind.argmax()]*0.5
            total_ind=np.zeros(searching_steps)



        if unormed_grad_list[-1] < stopping_cond:
            print(unormed_grad_list)
            return L @ L.T


def search_optimal_lr(L, S, k, search_s,range_max=1,min_range=0.1):
    Ls = []
    f_means = []
    lrs=[]
    for i, lr in enumerate(np.linspace(min_range, range_max, search_s) ):
        L, fval ,_,_= GD_ls(L, S, lr, k, iters=2)
        f_means.append(fval)
        Ls.append((L, lr))
        lrs.append(lr)
    f_means=np.array(f_means)
    min_id=f_means.argmin()
    L_opt, lr_opt = Ls[min_id][0], Ls[min_id][1]
    opt_ind=np.zeros(search_s)
    opt_ind[min_id]=1
    return L_opt, lr_opt , opt_ind , lrs


def GD_ls(L1, S1, lr, k, iters=1000, steps_counter=10):
    f_vals = []
    LS = []
    n = S1.shape[0]
    mask = np.tril(calculate_kband_mask(k, n))
    grads_size_list=[]
    for i in range(iters):
        gradient ,unorm_gard= calc_g_step(S1, L1, k, n, mask)
        L1 = L1 - lr * gradient
        L1 = stability_fix_L(L1)
        LS.append(L1)
        grads_size_list.append(np.abs(unorm_gard).sum())

        if iters-i<steps_counter:
            f_vals.append(calc_func(S1, L1))
    f_vals = np.array(f_vals)
    f_vals_mean = f_vals.mean()
    err_last=f_vals[-2]-f_vals[-1]
    return L1,f_vals_mean,err_last, grads_size_list






def init_random_L(lr, S, k, init_amount, iter_per_init):
    n = S.shape[0]
    fvals = np.zeros(init_amount)
    Ls = []
    for i in range(init_amount):
        L = generateRandomData(k, n)
        L_fin, fval,_,grad_size_list = GD_ls(L, S, lr, k, iter_per_init, steps_counter=3)
        fvals[i] = fval
        Ls.append(L_fin)
    L_opt = Ls[fvals.argmin()]
    return L_opt,grad_size_list

def calc_func(S, L):
    T_0 = (S).dot(L)
    functionValue = np.trace((T_0).dot(L.T))
    functionValue = functionValue - 2 * np.log(np.diag(L)).sum()
    return functionValue


def stability_fix_L(L1):
    indexes_on_diag = np.where(L1 < 0)
    for i, j in zip(indexes_on_diag[0], indexes_on_diag[1]):
        if i == j:
            L1[i, j] = STABILITY_EPS
    return L1

def synthize_values(n,k):
    S=generateRandomDataS(n)
    Opt_K=GD_total(S,k,10,iter_total=10000,iter_per_lr=500,learn_K=True)
    grad_vals=[]
    for i in range(2):
        grad_vals.append(GD_total(S,k,5,opt_K_flag=True,opt_K=Opt_K,return_grad=True,iter_per_lr=100))
    return grad_vals


def calc_stopping_crit(x):
    return np.log(x)*5.63996276e-05-1.14244167e-05

def calc_iter_step(x):
    return np.log(x)*100.23-327.938

def solve(S : np.ndarray,k : int):
    K=GD_solve(S,k,searching_steps=5,iter_total=10000000)
    return  K




