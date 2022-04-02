import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from scipy.stats import weibull_min
def timeit_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func_return_val = func(*args, **kwargs)
        end = time.perf_counter()
        print('{0:<10}.{1:<8} : {2:<8}'.format(func.__module__, func.__name__, end - start))
        return func_return_val

    return wrapper
color = sns.color_palette("tab10")
@timeit_wrapper
def np_weibull_paper_cdf(t, scale, shape):
    return 1 - np.exp(-((t / scale.T) ** shape.T))
@timeit_wrapper
def np_weibull_paper_pdf(t, scale, shape):
    return (shape / scale).T * (t ** ((shape - 1).T)) * np.exp(-((t / scale.T) ** shape.T))
@timeit_wrapper
def delta_threshold_rate(ts,freq,nfactor):
    return (ts*freq.T)**nfactor

shape = np.array([1.2567, 1.0727, 1.45316])
scale = np.array([82.1, 148, 89.2])

def Weibull_plot(shape=shape,scale=scale):
    # 可作為 模擬分析 power mosfet failure 的機率分布
    shape=shape.reshape(1, -1)
    scale=scale.reshape(1,-1)
    time_sample_point=1000
    x = np.linspace(0, 9E2, time_sample_point).reshape(1, -1)
    #
    cdf_ll = np_weibull_paper_cdf(t=x, shape=shape, scale=scale)
    pdf_ll = np_weibull_paper_pdf(t=x, shape=shape, scale=scale)

    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    sns.set(context='notebook', style='whitegrid')
    for idx, (cdf, pdf) in enumerate(zip(cdf_ll, pdf_ll)):
        # 第一章圖是用 CDF 反推回去 linear 的圖 詳細推倒請見筆記
        axes[0].plot(np.log(np.squeeze(x)[1:time_sample_point - 1]),np.log(-1 * np.log((1 - cdf)[1:time_sample_point - 1])),lw=2,label="shape=%.3f scale=%.1f" %(shape[0][idx],scale[0][idx]),color=color[idx])
        axes[1].plot(np.squeeze(x), cdf, lw=2, label="shape=%.3f scale=%.1f" %(shape[0][idx],scale[0][idx]), color=color[idx])
        axes[2].plot(np.squeeze(x), pdf, lw=2, label="shape=%.3f scale=%.1f" %(shape[0][idx],scale[0][idx]), color=color[idx])

    axes[0].set_title("Linear Relationship ", fontsize=20)
    axes[0].set_xlabel("Aging Time ln(t[min])", fontsize=15)
    axes[0].set_ylabel("ln[-ln(1-F(t))]", fontsize=15)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].legend()
    axes[0].yaxis.grid()

    axes[1].set_title("Failure Probability (CDF Graph)", fontsize=20)
    axes[1].set_xlabel("Aging Time (min)", fontsize=15)
    axes[1].set_ylabel("CDF F(t)", fontsize=15)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].legend()
    axes[1].yaxis.grid()

    axes[2].set_title("Failure Probability (PDF Graph)", fontsize=20)
    axes[2].set_xlabel("Aging Time (min)", fontsize=15)
    axes[2].set_ylabel("PDF F(t)", fontsize=15)
    axes[2].tick_params(axis='x', labelsize=15)
    axes[2].tick_params(axis='y', labelsize=15)
    axes[2].legend()
    axes[2].yaxis.grid()
    fig.suptitle('Weibull Distribution', fontsize=25)

    # ######################### verification part #由筆記 換算出 X value Y value 以及 跟scale 的關係 直接畫線型圖 與上面的plot 第一張圖一樣
    # for idx,(sh_s,sc_s) in enumerate(zip(shape.reshape(-1,1),scale.reshape(-1,1))):
    #     y_value=sh_s* np.log(x[0,1:])-(np.log(sc_s)*sh_s)
    #     axes[3].plot((np.log(np.squeeze(x)[1:])), np.squeeze(y_value), lw=2,label='paper regression No%d' % idx, color=color[idx])
    ##################################
    plt.show()

# ################ 計算 frequency 對於 lifetime 方面的評估
# cycle=1E12
# x = np.linspace(0, cycle, 10000).reshape(1, -1)
# fig2, axes2 = plt.subplots(1, 1, figsize=(20, 10))
# freq_ll=np.array([50E3,100E3,500E3]).reshape(1, -1)
# print(" {} equlivent to year time {}".format(freq_ll,(cycle/(freq_ll*60*60*24))))
# th_delta_ll=delta_threshold_rate(ts=x,freq=freq_ll,nfactor=0.05)
# for idx, (th_delta,freq) in enumerate (zip(th_delta_ll,freq_ll.reshape(-1,1))):
#     axes2.plot(np.squeeze(x),th_delta,lw=2,label='freq : %d' % freq[0],color=color[idx])
# axes2.legend()
# axes2.yaxis.grid()
# plt.show()
# #########################

sim_num=1000
ag_th=0.8
ag_ll=[]
x = weibull_min.rvs(shape[0],scale=scale[0],loc=0,size=sim_num)
print(x)
Weibull_plot(shape=shape,scale=scale)