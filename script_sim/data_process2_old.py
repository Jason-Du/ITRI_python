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
def anlze_log_file(meas_params=[],aging_coffs=[],freq_coff="freq",logfile="",tol_range=[0,None],freq_sel=None):
    # range 的設計是 為了縮小繪圖的區間範圍 若 step 是 100 simulation 0.01 per step 則 [0,50] 可以模擬出 1~1.5 倍的模擬結果 預設是全部print 出
    # freq_sel = None 時 與正常使用時一樣
    data={}
    data = logfile_reader(logfile)
    #定義要觀察的量測指標
    data_dict={}
    step_dict = {}
    ref = {}
    freq_idxs=[]
    if freq_sel is not None:
        freq_idxs = [i for i in range(len(data[freq_coff])) if data[freq_coff][i] == freq_sel]
    for aging_coff in aging_coffs:
        if freq_sel is not None:#若是log檔內包含多種頻率 需要進行資料處理 freq coff 有兩種type freq 或是 r_freq
            step_dict[aging_coff] =[float("%.2f"%x) for x in list(itemgetter(*freq_idxs)(data[aging_coff]))[tol_range[0]:tol_range[1]]]
        else :
            step_dict[aging_coff] = [float("%.2f" % x) for x in data[aging_coff][tol_range[0]:tol_range[1]]]  # 因為在spice 的模擬過程中 1 step 為 0.01 精度故調整為0.2f
    ref_ll=[]
    for aging_coff in aging_coffs:
        ref_ll=ref_ll+[ i for i in range(len(step_dict[aging_coff])) if step_dict[aging_coff][i]==0]
    ref_idx=max(set(ref_ll), key=ref_ll.count)
    for meas_param in meas_params:
        if freq_sel is not None:
            data_dict[meas_param]=list(itemgetter(*freq_idxs)(data[meas_param]))[tol_range[0]:tol_range[1]]
            ref[meas_param]=data_dict[meas_param][ref_idx]
        else:
            data_dict[meas_param]=data[meas_param][tol_range[0]:tol_range[1]]
            ref[meas_param]=data_dict[meas_param][ref_idx]
    # duty ratio 需要特別處理 因為 spice 可能抓到 的是 duty on ratio 或是 1+duty_on ratio 查看 spice script 檔能有所理解
    if "duty_ratio"  in meas_params:
        data_dict["duty_ratio"]=[i if i<=1 else i-1 for i in data_dict["duty_ratio"] ]
        ref["duty_ratio"] = data_dict["duty_ratio"][ref_idx]-1 if data_dict["duty_ratio"][ref_idx]>1else data_dict["duty_ratio"][ref_idx]
    var_dict = {}
    var_max_dict = {}
    var_max_idxs = {}
    var_dict_percent={}
    #以第一筆資料當作比較依據 後需資料除以 第一筆資料 查看其變化幅度 <1 為減少 >1 為增加
    for meas_param in meas_params:
        data_dict[meas_param]= [float(x/ref[meas_param]) for x in data_dict[meas_param]]
        var_dict[meas_param]=[ x-1 for x in data_dict[meas_param]]    #資料全部減1 獲取變化的"幅度" <0 為減少 >0 為增加
        var_dict_percent[meas_param] = [x*100 for x in var_dict[meas_param]]
        var_max_dict[meas_param]=max(var_dict[meas_param],key=abs) #找出變化最大變化的"幅度"
        var_max_idxs[meas_param]=var_dict[meas_param].index(var_max_dict[meas_param])#    # 把變化最大的資料 的index 儲存起來
    # 把變化最大的資料 print 出來
    # print(" rise time increase rate  : {} %".format( (  var_max_dict["tr1"]*100) )  )
    # print(" fall time increase rate  : {} %".format( ( var_max_dict["tr2"]*100) )  )
    # print(" maximum id current  increase rate : {} % ".format( ( var_max_dict["imax"]*100) )  )

    return data_dict,step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict

######### Draw1 分析不同頻率下的狀況 ########Frequency X 軸 Y 軸 取變化量最大者##########################
# column_data = list(zip(*data_collections))
# color = sns.color_palette("Set1")
# freq_ll_k = [x / 1E3 for x in freq_ll]
# fig2, axes2 = plt.subplots(1, len(meas_params), figsize=(20, 10))
# for idx, (indicator, title) in enumerate(zip(meas_params, title_ll)):
#     var_idx_ll = [x[indicator] for x in column_data[5]]
#     var_percnt_ll = [percent_dict[indicator][var_idx] for var_idx, percent_dict in zip(var_idx_ll, column_data[4])]
#
#     var_percnt_ll = [round((x / var_percnt_ll[0] - 1) * 100, 2) for x in var_percnt_ll]
#
#     axes2[idx].plot(freq_ll_k, var_percnt_ll, color=color[idx])
#     axes2[idx].yaxis.set_major_locator(MaxNLocator(5))
#     axes2[idx].xaxis.set_major_locator(MaxNLocator(5))
#     axes2[idx].set_xlabel("Frequency (kHz)", fontsize=15)
#     axes2[idx].set_title(title, fontsize=20)
#     axes2[idx].tick_params(axis='x', labelsize=15)
#     axes2[idx].tick_params(axis='y', labelsize=15)
#     axes2[idx].xaxis.grid()
# plt.show()