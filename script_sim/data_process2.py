from PyLTSpice.LTSteps import LTSpiceLogReader
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
import matplotlib.ticker as tick
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from os import listdir
from palettable.colorbrewer import BrewerMap
from os.path import isfile, isdir, join
import re
import seaborn as sns
import pandas as pd
import statistics
from mpl_toolkits import mplot3d
from matplotlib import cm
import numpy as np

DRAW_1=False
DRAW_2=True
MAX_VARIATION_ANALY=False
param_num=1
def anlze_log_file(meas_params=[],aging_coffs=[],logfile=""):
    # "./result/r8002cnd3_MOS_N_L.log"
    # "monte_SiC.log"
    # print("Number of steps  :", data.step_count)
    data = LTSpiceLogReader(logfile)
    step_names = data.get_step_vars()
    meas_names = data.get_measure_names()
    # print(step_names)# for step in step_names  print(step)# tol
    # print(meas_names)#data[step][i])for step in step_names] data[name][i])for name in meas_names])
    # print(data["tol1"])
    #定義要觀察的量測指標
    data_dict={}
    step_dict = {}
    ref = {}
    for meas_param in meas_params:
        data_dict[meas_param]=data[meas_param]
        ref[meas_param]=statistics.median(data_dict[meas_param])
    for aging_coff in aging_coffs:
        step_dict[aging_coff]=[float("%.2f"%x) for x in data[aging_coff]]

    var_dict = {}
    var_max_dict = {}
    var_max_idxs = {}
    var_dict_percent={}
    #以第一筆資料當作比較依據 後需資料除以 第一筆資料 查看其變化幅度 <1 為減少 >1 為增加
    for meas_param in meas_params:
        data_dict[meas_param]= [float(x/ref[meas_param]) for x in data_dict[meas_param]]
        var_dict[meas_param]=[ x-1 for x in data_dict[meas_param]]    #資料全部減1 獲取變化的"幅度" <0 為減少 >0 為增加
        var_dict_percent[meas_param] = [x*100for x in var_dict[meas_param]]
        var_max_dict[meas_param] =max(var_dict[meas_param],key=abs) #找出變化最大變化的"幅度"
        var_max_idxs[meas_param]=var_dict[meas_param].index(var_max_dict[meas_param])#    # 把變化最大的資料 的index 儲存起來

    # 把變化最大的資料 print 出來
    # print(" rise time increase rate  : {} %".format( (  var_max_dict["tr1"]*100) )  )
    # print(" fall time increase rate  : {} %".format( ( var_max_dict["tr2"]*100) )  )
    # print(" maximum id current  increase rate : {} % ".format( ( var_max_dict["imax"]*100) )  )

    return data_dict,step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict
if __name__ == '__main__':
    if MAX_VARIATION_ANALY:
        files = listdir("./result/{}".format(param_num))
        meas_params = ["tr1", "tr2", "imax"]
        aging_coffs = ["tol1"]
        var_max_idxs={}
        one_max_variation={}
        max_variations={"measure_value":[],"measure_name":[],"param_name":[]}

        for f_idx, f in enumerate(files):
            pass

            logfile = "./result/{}/".format(param_num)+f
            data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict= anlze_log_file(meas_params=meas_params,aging_coffs=aging_coffs,logfile=logfile)
            pattern = re.compile(r"r8002cnd3_(.*).log")
            mod_par=re.search(pattern,f)
            # if [var_max_dict[meas_param]for meas_param in meas_params]!=[0,0,0]:#納入繪圖條件設置[0,0,0]代表三個量測值皆無變化不與討論
            if var_max_dict["imax"]!=0 and mod_par.group(1) not in ["MOS_N_L","MOS_N_W"]:#濾出 imax 不等於0的資訊
                max_variations["measure_value"]=max_variations["measure_value"]+[var_max_dict[meas_param]*100 for meas_param in meas_params]
                max_variations["measure_name"]=max_variations["measure_name"]+[meas_param for meas_param in meas_params]
                max_variations["param_name"]=max_variations["param_name"]+[mod_par.group(1)for meas_param in meas_params]

        df=pd.DataFrame(max_variations)

        one_max_variation["measure_value"] = [x for x,m_name in zip(max_variations["measure_value"],max_variations["measure_name"]) if m_name == "imax"]
        one_max_variation["param_name"] = [x for x,m_name in zip(max_variations["param_name"],max_variations["measure_name"]) if m_name == "imax"]
        df2 = pd.DataFrame(one_max_variation)
        df2=df2.sort_values("measure_value")


        fig, ax = plt.subplots(figsize=(20, 10))
        # color=sns.color_palette("Set2")
        # sns.barplot(x="param_name",y="measure_value",hue="measure_name",data=df,ci=None,ax=ax,palette=color[0:3])
        # ax.set_ylim(-1,1.5)
        # ax.yaxis.grid()


        palette_2 = sns.color_palette("flare", n_colors=len(one_max_variation["param_name"]))
        # palette_2.reverse()
        ax_s=sns.barplot(x="param_name", y="measure_value", data=df2, ci=None, ax=ax,palette=palette_2)#palette 讓使用者可以定義多個顏色在圖表上 color 選單僅有一個參數可以使用
        ax.set_ylabel("rate (%)", fontsize=15, rotation=0)
        ax.yaxis.set_label_coords(-0.05, 1.02)
        ax.yaxis.grid()
        # ax.set_yscale('log')
        for p in ax_s.patches:
            ax_s.annotate("%.3f" % p.get_height()+"%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),textcoords='offset points')
        ax.set_xlabel("Aging parameter", fontsize=20)
        _, xlabels = plt.xticks()
        ax_s.set_xticklabels(xlabels, size=12)
        ax_s.set_yticklabels(ax_s.get_yticks(), size=15)
        ax.set_title(" Max(ID) variation v.s. Aging parameter", fontsize=20)
        plt.show()





    # 繪製曲線圖 橫縱軸都是以增加幅度繪製
    if DRAW_2:
        logfile = "monte_SiC.log"
        meas_params = ["tr1", "tr2", "imax"]
        aging_coffs = ["tol1","tol2"]
        data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict = anlze_log_file(meas_params=meas_params, aging_coffs=aging_coffs, logfile=logfile)
        pass
        palette_2 = sns.color_palette("rocket", as_cmap=True)
        fig, ax = plt.subplots(1,1,figsize=(20, 10))
        one_variation={}
        one_variation["imax"]=var_dict_percent["imax"]
        one_variation["tol1"] =step_dict["tol1"]
        one_variation["tol2"] =step_dict["tol2"]
        one_variation=pd.DataFrame(one_variation)
        one_variation=one_variation.pivot("tol1","tol2","imax")
        ax_s = sns.heatmap(one_variation,xticklabels=10,yticklabels=10)
        ax_s.set_xticklabels(ax_s.get_xmajorticklabels(), fontsize = 12)
        ax_s.set_yticklabels(ax_s.get_ymajorticklabels(), fontsize = 12)
        ax.set_xlabel("MOS_N VTO variation(rate)", fontsize=15)
        ax.set_ylabel("MOS_N GAMMA variation(rate)", fontsize=15)
        ax.set_title("Max(ID) variation v.s. MOS_N GAMMA / MOS_N VTO",fontsize=20)
        plt.show()






    if DRAW_1:
        logfile = "./result/1/r8002cnd3_MOS_N_CGSO.log"
        pattern = re.compile(r"r8002cnd3_(.*).log")
        mod_par = re.search(pattern,logfile)
        meas_params = ["tr1", "tr2", "imax"]
        aging_coffs = ["tol1"]
        data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict=anlze_log_file(meas_params=meas_params,aging_coffs=aging_coffs,logfile=logfile)
        step_list=[x+1for x in step_dict["tol1"]]



        y_low=-100
        y_high=100
        fig,axes=plt.subplots(1,3,figsize=(20, 10))
        #繪製模擬時間內的最大電流
        color = sns.color_palette("Set1")
        # sns.set()
        axes[0].plot(step_list,var_dict_percent["imax"],color=color[0] )
        axes[0].plot(step_list[var_max_idxs["imax"]],var_dict_percent["imax"][var_max_idxs["imax"]],marker="*",color="black",markersize=10)
        axes[0].text(step_list[var_max_idxs["imax"]], var_dict_percent["imax"][var_max_idxs["imax"]], "%.3f"%var_dict_percent["imax"][var_max_idxs["imax"]]+" %\n\n", fontsize=12, color="k", style="italic", weight="bold", verticalalignment='center',horizontalalignment='right', rotation=0)
        axes[0].yaxis.set_major_locator(MaxNLocator(5))
        axes[0].xaxis.set_major_locator(MaxNLocator(5))
        axes[0].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
        axes[0].set_title("Max(drain current) (%)",fontsize=20)
        axes[0].tick_params(axis='x', labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)
        # axes[0].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
        axes[0].set_xlim(1.00,2.00)
        axes[0].yaxis.grid()
        axes[0].set_ylim(y_low,y_high)
        #繪製模擬時間內的rise time
        axes[1].plot(step_list,var_dict_percent["tr1"],color=color[1])
        axes[1].plot(step_list[var_max_idxs["tr1"]],var_dict_percent["tr1"][var_max_idxs["tr1"]],marker="*",color="black",markersize=10)
        axes[1].text(step_list[var_max_idxs["tr1"]], var_dict_percent["tr1"][var_max_idxs["tr1"]],"%.3f" % var_dict_percent["tr1"][var_max_idxs["tr1"]] + " %\n\n", fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        axes[1].yaxis.set_major_locator(MaxNLocator(5))
        axes[1].xaxis.set_major_locator(MaxNLocator(5))
        axes[1].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
        axes[1].set_title("rise time (%) ",fontsize=20)
        axes[1].tick_params(axis='x', labelsize=15)
        axes[1].tick_params(axis='y', labelsize=15)
        # axes[1].set_ylim(min(data_dict["tr1"]),max(data_dict["tr1"]))
        axes[1].yaxis.grid()
        axes[1].set_ylim(y_low,y_high)
        #繪製模擬時間內的fall time
        axes[2].plot(step_list,var_dict_percent["tr2"],color=color[2] )
        axes[2].plot(step_list[var_max_idxs["tr2"]],var_dict_percent["tr2"][var_max_idxs["tr2"]],marker="*",color="black",markersize=10)
        axes[2].text(step_list[var_max_idxs["tr2"]], var_dict_percent["tr2"][var_max_idxs["tr2"]],"%.3f" % var_dict_percent["tr2"][var_max_idxs["tr2"]] + " %\n\n", fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        axes[2].yaxis.set_major_locator(MaxNLocator(5))
        axes[2].xaxis.set_major_locator(MaxNLocator(5))
        axes[2].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
        axes[2].set_title("fall time (%)",fontsize=20)
        axes[2].tick_params(axis='x', labelsize=15)
        axes[2].tick_params(axis='y', labelsize=15)
        # axes[2].set_ylim(min(data_dict["tr2"]),max(data_dict["tr2"]))
        axes[2].yaxis.grid()
        axes[2].set_ylim(y_low,y_high)

        plt.show()