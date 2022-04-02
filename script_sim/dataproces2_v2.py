import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import listdir
import re
import seaborn as sns
import pandas as pd
import numpy as np
from logfile_process import logfile_reader
from matplotlib.ticker import FormatStrFormatter
from operator import itemgetter
DRAW_1=True
DRAW_2=False
MAX_VARIATION_ANALY=False
CIRCUIT_ANALY=False
CIRCUIT_Single=False#與# CIRCUIT_ANALY 是綁定一起的
import time
param_num=1
from collections import Counter
# Device_list
# rq3p300bh
# rsj400n10
# "rj1l12bgn"
# "rd3l050sn"
# "rcj510n25"
# "rcj700N20"
# "all_freq"
# "rj1p12bbd"
device_name="rsj400n10"#DRAW_1 MAX_VARIATION_ANALY CIRCUIT_ANALY 皆會用到
device_condition="all_freq_7.5Vg_40Vds_70A" #全部都會用到
circuit_condition="all_freq"  #CIRCUIT_ANALY 會用到
# "all_freq"
# "Rk_60kHz"
#"40khz_7.5Vg_40Vds_1.6R"
# "all_freq_7.5Vg_40Vds_70A"
R_freq_dict={
    #左邊為 頻率 右邊為電阻
    200000:120000,
    150000:155000,
    100000:250000,
    80000:300000,
    60000:400000,
    None:None
}
if DRAW_1:
    logfile = "./result/{}/{}/1/{}_MOS_N_VTO.log".format(device_name, device_condition, device_name)
    pattern = re.compile(r"{}_(.*).log".format(device_name))
    mod_par = re.search(pattern, logfile)
    meas_params = ["tr1", "tr2", "imax", "vth"]
    title_ll = ["rise time (%)", "fall time (%)", "Ids (%)", "Vth(%)"]
    aging_coffs = ["tol1"]
    freq_sel = 200000  # 若為單一頻率架購則設為None
    data = logfile_reader(logfile)
    df = pd.DataFrame(data)
    if freq_sel is not None:
        df_freq=df[(df.freq == freq_sel)].columns.intersection(meas_params)
    else:
        df_freq=df.columns.intersection(meas_params)

    # fig, axes = plt.subplots(1, len(meas_params), figsize=(20, 10))
    # #繪製模擬時間內的最大電流 Rise time fall time
    # color = sns.color_palette("Set1")
    # for idx, (indicator, title) in enumerate(zip(meas_params, title_ll)):
    #     data_ary=np.array(df_data[indicator])
    #     data_percent=(data_ary/)-1
    #     y_low = min(var_dict_percent[indicator]) - 5
    #     y_high = max(var_dict_percent[indicator]) + 5
    #     axes[idx].plot(step_list, var_dict_percent[indicator], color=color[idx])
    #     axes[idx].plot(step_list[var_max_idxs[indicator]], var_dict_percent[indicator][var_max_idxs[indicator]],
    #                    marker="*", color="black", markersize=10)
    #     axes[idx].text(step_list[var_max_idxs[indicator]], var_dict_percent[indicator][var_max_idxs[indicator]],
    #                    "\n\n%.2f" % var_dict_percent[indicator][var_max_idxs[indicator]] + "%\n\n", fontsize=12,
    #                    color="k", style="italic", weight="bold", verticalalignment='center',
    #                    horizontalalignment='right', rotation=0)
    #     axes[idx].yaxis.set_major_locator(MaxNLocator(5))
    #     axes[idx].xaxis.set_major_locator(MaxNLocator(5))
    #     axes[idx].set_xlabel(mod_par.group(1) + " variation(rate)", fontsize=15)
    #     axes[idx].set_title(title, fontsize=20)
    #     axes[idx].tick_params(axis='x', labelsize=15)
    #     axes[idx].tick_params(axis='y', labelsize=15)
    #     # axes[idx].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
    #     axes[idx].set_xlim(1.00, max(step_list))
    #     axes[idx].yaxis.grid()
    #     axes[idx].set_ylim(y_low, y_high)
    # plt.show()