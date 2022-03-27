from PyLTSpice.LTSteps import LTSpiceLogReader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import listdir
import re
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
DRAW_1=False
DRAW_2=False
MAX_VARIATION_ANALY=False
CIRCUIT_ANALY=True
param_num=1
device_name="rcj510n25"
# "rd3l050sn"
test_condition="10Vg_40Vds_5.5R"

def anlze_log_file(meas_params=[],aging_coffs=[],logfile="",tol_range=[0,50]):
    # range 的設計是 為了縮小繪圖的區間範圍 若 step 是 100 simulation 0.01 per step 則 [0,50] 可以模擬出 1~1.5 倍的模擬結果
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
    for aging_coff in aging_coffs:
        step_dict[aging_coff]=[float("%.2f"%x) for x in data[aging_coff][tol_range[0]:tol_range[1]]]#因為在spice 的模擬過程中 1 step 為 0.01 精度故調整為0.2f

    step_size=len(step_dict[aging_coff])
    step_dict_bool = np.array([True] *step_size)
    for aging_coff in aging_coffs:
        step_dict_bool= step_dict_bool & np.array([float(i)==0.0 for i in step_dict[aging_coff][tol_range[0]:tol_range[1]]])
        # print(step_dict_bool)
    ref_idx=np.where(step_dict_bool == True)

    for meas_param in meas_params:
        data_dict[meas_param]=data[meas_param][tol_range[0]:tol_range[1]]
        ref[meas_param]=data_dict[meas_param][ref_idx[0][0]]
    # duty ratio 需要特別處理 因為 spice 可能抓到 的是 duty on ratio 或是 1+duty_on ratio 查看 spice script 檔能有所理解
    if "duty_ratio"  in meas_params:
        data_dict["duty_ratio"]=[i if i<=1 else i-1 for i in data_dict["duty_ratio"] ]

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
if __name__ == '__main__':
    if CIRCUIT_ANALY:
        # 在分析的過程中 若今天"rcj510n25" device 取 VTO 為影響 threshold voltage 的主要因素 下去模擬 25step 0.01 rate/step
        # switching loss 在 第7 次模擬結果後 升的超級高(第 10 step) 原因是因為站空比拉大 (90%) P_Sw 的 spice 的 script 裡面是 V*I 沒有足夠的時間下降 導製 ZVS ZCS 失去效用
        # switching loss (第 10 step) 之後開始劇降 原因是因為 output voltage 開始降低 拉不到 規格中的 42 V
        #會發現系統後半段 的模擬 efficiency掉得很快 是因為 電壓已經拉不上去 拉不到 規格中的 42 V
        # 原因是因為沒掉穩太狀態 所以能參考的資料 在模擬時間有限的狀況下(10ms) 僅能參考 7step 的資料資訊
        # set time (第 10 step) 之前 呈現 EXPONENTIAL GRAPH 式 的增長 driving 的能力變弱了 上升期 階段的流經電流慢慢隨著老化在下降
        # 上述原因導致了 需要比較長的時間來做 set up 進一步導致 diode 損壞 I peak avg break
        # duty ratio 在前面7 step 模擬的 variation 算是可以接受
        # ripple voltage 不適合拿來做為 一個 因為太過浮動 但當系統崩快時 output voltage 開始降低 拉不到 規格中的 42 V 也算是崩快的一種依據
        # 指標警示 可選擇 set up time 指標警示系統 可訂 530%
        # Switching 作為 指標警示系統 可訂 6.37%
        # efficiency 在前面9個step 變化不明顯 是因為 系統利用 條大佔空比 去彌補 trade -off switching loss 增加


        meas_params = ["p_sw","voutpp","eff","t_set","duty_ratio","vout_avg"]
        aging_coffs = ["tol1"]
        title_ll = ["switching loss (%)", "ripple voltage (%)", "efficiency(%)","set up time (%)","switching on\nduty ratio (%)","output voltage(%)"]
        tol_range1= [0,9]
        data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict=anlze_log_file(meas_params=meas_params,
                                                                                                aging_coffs=aging_coffs,
                                                                                                logfile="./result/LTC1871-7_F09.log",
                                                                                                tol_range=tol_range1)
        # ########################將 step 的 param variation 轉為觀測用 的指標 VTO (%) -->threshold voltage (%)
        meas_params2 = ["tr1", "tr2", "imax","vth"]
        aging_coffs2 = ["tol1"]
        param_sel = "vth"
        tol_range2 =[0,50]
        logfile2 = "./result/{}/{}/1/{}_MOS_N_VTO.log".format(device_name, test_condition, device_name)
        data_dict2, step_dict2,var_dict2,var_dict_percent2,var_max_idxs2,var_max_dict2=anlze_log_file(meas_params=meas_params2,
                                                                                                aging_coffs=aging_coffs2,
                                                                                                logfile=logfile2,
                                                                                                tol_range=tol_range2
                                                                                                )
        # ######################################

        step_list = var_dict_percent2[param_sel][tol_range1[0]:tol_range1[1]]
        y_low = min([x for k in var_dict_percent.values() for x in k]) - 5
        y_high = max([x for k in var_dict_percent.values() for x in k]) + 5
        fig, axes = plt.subplots(1, len(meas_params), figsize=(20, 10))
        sns.set(context='notebook', style='whitegrid')
        for idx, (indicator, title) in enumerate(zip(meas_params, title_ll)):
            color = sns.color_palette("tab10")
            y_low = min(var_dict_percent[indicator]) - 5
            y_high = max(var_dict_percent[indicator]) + 5

            axes[idx].plot(step_list, var_dict_percent[indicator], color=color[idx])
            axes[idx].plot(step_list[var_max_idxs[indicator]], var_dict_percent[indicator][var_max_idxs[indicator]],
                           marker="*", color="black", markersize=10)
            axes[idx].text(step_list[var_max_idxs[indicator]], var_dict_percent[indicator][var_max_idxs[indicator]],
                           "\n\n%.2f" % var_dict_percent[indicator][var_max_idxs[indicator]] + "%\n\n", fontsize=12,
                           color="k", style="italic", weight="bold", verticalalignment='center',
                           horizontalalignment='right', rotation=0)
            axes[idx].yaxis.set_major_locator(MaxNLocator(5))
            axes[idx].xaxis.set_major_locator(MaxNLocator(5))
            axes[idx].set_xlabel("Vth" + " variation(%)", fontsize=15)
            axes[idx].set_title(title, fontsize=20)
            axes[idx].tick_params(axis='x', labelsize=15)
            axes[idx].tick_params(axis='y', labelsize=15)
            # axes[idx].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
            axes[idx].set_xlim(0.00, max(step_list))
            axes[idx].yaxis.grid()
            axes[idx].set_ylim(y_low, y_high)
        plt.show()

    if MAX_VARIATION_ANALY:
        files = listdir("./result/{}/{}/{}".format(device_name,test_condition,param_num))
        meas_params = ["tr1","tr2","imax","vth"]
        aging_coffs = ["tol1"]
        var_max_idxs={}
        one_max_variation={}
        max_variations={"measure_value":[],"measure_name":[],"param_name":[]}
        param_sel="vth"

        for f_idx, f in enumerate(files):
            pass
            print(f)
            logfile = "./result/{}/{}/{}/".format(device_name,test_condition,param_num)+f
            data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict= anlze_log_file(meas_params=meas_params,
                                                                                                     aging_coffs=aging_coffs,
                                                                                                     logfile=logfile,
                                                                                                     tol_range=[0,50]
                                                                                                     )
            pattern = re.compile(r"%s_(.*).log"%device_name)
            mod_par=re.search(pattern,f)
            # if [var_max_dict[meas_param]for meas_param in meas_params]!=[0,0,0]:#納入繪圖條件設置[0,0,0]代表三個量測值皆無變化不與討論
            if var_max_dict[param_sel]!=0 and mod_par.group(1) not in ["MOS_N_L","MOS_N_W"]:#濾出 imax 不等於0的資訊 撇除不要考慮的 parameter
                max_variations["measure_value"]=max_variations["measure_value"]+[var_max_dict[meas_param]*100 for meas_param in meas_params]
                max_variations["measure_name"]=max_variations["measure_name"]+[meas_param for meas_param in meas_params]
                max_variations["param_name"]=max_variations["param_name"]+[mod_par.group(1)for meas_param in meas_params]

        df=pd.DataFrame(max_variations)

        one_max_variation["measure_value"] = [round(x,3) for x,m_name in zip(max_variations["measure_value"],max_variations["measure_name"]) if m_name == param_sel]
        one_max_variation["param_name"] = [x for x,m_name in zip(max_variations["param_name"],max_variations["measure_name"]) if m_name == param_sel]

        df2 = pd.DataFrame(one_max_variation)
        df2=df2.sort_values("measure_value")


        fig, ax = plt.subplots(figsize=(20, 10))
        #化出tr1 tr2 id current 的大小分部
        # color=sns.color_palette("Set2")
        # sns.barplot(x="param_name",y="measure_value",hue="measure_name",data=df,ci=None,ax=ax,palette=color[0:3])
        # ax.set_ylim(-1,1.5)
        # ax.yaxis.grid()

        # 畫出前面限定條件下 EX只抓取 電流發生變化者 排除W L parameter 者 來做分析
        palette_2 = sns.color_palette("flare", n_colors=len(one_max_variation["param_name"]))# imax 下面有幾個param 是有影響的
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
        ax_s.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_title(param_sel+" v.s. Aging parameter", fontsize=20)
        plt.show()


    # 繪製曲線圖 橫縱軸都是以增加幅度繪製
    if DRAW_2:
        # "monte_SiC_rd31050sn.log"
        #
        logfile ="VTO_GAMMA_r8002cnd3.log"
        meas_params = ["tr1", "tr2", "imax"]
        aging_coffs = ["tol1","tol2"]
        title_ll = ["rise time (%)", "fall time (%)", "Ids (%)", "Vth(%)"]
        threshold_value=11# 以5來說 是5% 決定後續繪圖 +- 5 5 的範圍
        sel_param="imax"
        inject_param=["VTO","GAMMA  "]
        data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict = anlze_log_file(meas_params=meas_params,
                                                                                                      aging_coffs=aging_coffs,
                                                                                                      logfile=logfile,
                                                                                                      tol_range=[0,100]
                                                                                                      )

        palette_1=sns.color_palette("hls", 8)
        fig, ax = plt.subplots(1,1,figsize=(20, 10))
        one_variation={}
        one_variation[sel_param]=var_dict_percent[sel_param]

        one_variation["tol1"] =[x+1for x in step_dict["tol1"]]
        one_variation["tol2"] =[x+1for x in step_dict["tol2"]]

        one_variation=pd.DataFrame(one_variation)
        one_variation=one_variation.pivot(columns ="tol2",index="tol1",values=sel_param) #df.value 為二為矩陣 先是column 資料排完才會再往下一條row 進行排列資料 df.values[0] 為一整條ROW的資料

        rows, cols=np.where(one_variation==0)#return  原點 index
        ax.plot(rows,cols,marker="*",color="black", markersize=10)#標記原點
        rows,cols = np.where( (threshold_value>=one_variation) & (one_variation>=(-threshold_value)) )  # return   +- threhold 範圍的點
        for row, col in zip(rows, cols):
            ax.plot(col, row, marker=".", color="red", markersize=2)#畫出+- threhold 範圍的點
        plot_s=[[row,col] for row,col in zip(rows,cols)]# +- threhold 範圍的點合成矩陣
        col_set=np.array(list(set(cols)))#扣除重複出現的col 目的是為了要求邊界
        row_top=np.array(list(map(lambda i : max([plot_[0] for plot_ in plot_s if plot_[1]==i]),col_set))) #在同一col 下 row 的大小值為其邊界點
        row_ground = np.array(list(map(lambda i: min([plot_[0] for plot_ in plot_s if plot_[1] == i]), col_set)))
        ax.plot(col_set, row_top,color=palette_1[0])
        ax.plot(col_set, row_ground,color=palette_1[1])

        palette_2 = sns.color_palette("rocket", as_cmap=True)
        ax_s = sns.heatmap(one_variation,xticklabels=10,yticklabels=10,cmap="YlGnBu",cbar_kws={"orientation":"vertical","label":"%s variation"%title_ll[2],"location":"right",'ticks': [40,20,5, 0, -5, -20,-40, -60,-80]})
        ax_s.set_xticklabels(ax_s.get_xmajorticklabels(), fontsize = 15)
        ax_s.set_yticklabels(ax_s.get_ymajorticklabels(), fontsize = 15)
        ax_s.figure.axes[-1].yaxis.label.set_size(18)
        cbar =ax_s.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        ax.set_ylabel("%s variation(rate)"%inject_param[0], fontsize=18)
        ax.set_xlabel("%s variation(rate)"%inject_param[1], fontsize=18)
        ax.set_title(" {} variation v.s. {} / {} Variation".format(title_ll[2],inject_param[0],inject_param[1]),fontsize=20)
        plt.show()



    if DRAW_1:
        logfile = "./result/{}/{}/1/{}_MOS_N_VTO.log".format(device_name,test_condition,device_name)
        pattern = re.compile(r"{}_(.*).log".format(device_name))
        mod_par = re.search(pattern,logfile)
        meas_params = ["tr1", "tr2", "imax","vth"]
        title_ll=["rise time (%)","fall time (%)","Ids (%)","Vth(%)"]
        aging_coffs = ["tol1"]

        data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict=anlze_log_file(meas_params=meas_params,
                                                                                                aging_coffs=aging_coffs,
                                                                                                logfile=logfile,
                                                                                                tol_range=[0,50])
        step_list=[x+1for x in step_dict[aging_coffs[0]]]

        y_low=min([x for k in var_dict_percent.values() for x in k ])-5
        y_high=max([x for k in var_dict_percent.values() for x in k ])+5

        fig,axes=plt.subplots(1,len(meas_params),figsize=(20, 10))
        for idx,(indicator,title) in enumerate(zip(meas_params,title_ll)):
            color = sns.color_palette("Set1")

            axes[idx].plot(step_list,var_dict_percent[indicator],color=color[idx] )
            axes[idx].plot(step_list[var_max_idxs[indicator]],var_dict_percent[indicator][var_max_idxs[indicator]],marker="*",color="black",markersize=10)
            axes[idx].text(step_list[var_max_idxs[indicator]], var_dict_percent[indicator][var_max_idxs[indicator]], "\n\n%.2f"%var_dict_percent[indicator][var_max_idxs[indicator]]+"%\n\n", fontsize=12, color="k", style="italic", weight="bold", verticalalignment='center',horizontalalignment='right', rotation=0)
            axes[idx].yaxis.set_major_locator(MaxNLocator(5))
            axes[idx].xaxis.set_major_locator(MaxNLocator(5))
            axes[idx].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
            axes[idx].set_title(title,fontsize=20)
            axes[idx].tick_params(axis='x', labelsize=15)
            axes[idx].tick_params(axis='y', labelsize=15)
            # axes[idx].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
            axes[idx].set_xlim(1.00,max(step_list))
            axes[idx].yaxis.grid()
            axes[idx].set_ylim(y_low,y_high)

        # #繪製模擬時間內的最大電流
        # color = sns.color_palette("Set1")
        # # sns.set()
        # axes[0].plot(step_list,var_dict_percent["imax"],color=color[0] )
        # axes[0].plot(step_list[var_max_idxs["imax"]],var_dict_percent["imax"][var_max_idxs["imax"]],marker="*",color="black",markersize=10)
        # axes[0].text(step_list[var_max_idxs["imax"]], var_dict_percent["imax"][var_max_idxs["imax"]], "%.3f"%var_dict_percent["imax"][var_max_idxs["imax"]]+" %\n\n", fontsize=12, color="k", style="italic", weight="bold", verticalalignment='center',horizontalalignment='right', rotation=0)
        # axes[0].yaxis.set_major_locator(MaxNLocator(5))
        # axes[0].xaxis.set_major_locator(MaxNLocator(5))
        # axes[0].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
        # axes[0].set_title("Drain current (%)",fontsize=20)
        # axes[0].tick_params(axis='x', labelsize=15)
        # axes[0].tick_params(axis='y', labelsize=15)
        # # axes[0].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
        # axes[0].set_xlim(1.00,2.00)
        # axes[0].yaxis.grid()
        # axes[0].set_ylim(y_low,y_high)
        # #繪製模擬時間內的rise time
        # axes[1].plot(step_list,var_dict_percent["tr1"],color=color[1])
        # axes[1].plot(step_list[var_max_idxs["tr1"]],var_dict_percent["tr1"][var_max_idxs["tr1"]],marker="*",color="black",markersize=10)
        # axes[1].text(step_list[var_max_idxs["tr1"]], var_dict_percent["tr1"][var_max_idxs["tr1"]],"%.3f" % var_dict_percent["tr1"][var_max_idxs["tr1"]] + " %\n\n", fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        # axes[1].yaxis.set_major_locator(MaxNLocator(5))
        # axes[1].xaxis.set_major_locator(MaxNLocator(5))
        # axes[1].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
        # axes[1].set_title("rise time (%) ",fontsize=20)
        # axes[1].tick_params(axis='x', labelsize=15)
        # axes[1].tick_params(axis='y', labelsize=15)
        # # axes[1].set_ylim(min(data_dict["tr1"]),max(data_dict["tr1"]))
        # axes[1].yaxis.grid()
        # axes[1].set_ylim(y_low,y_high)
        # #繪製模擬時間內的fall time
        # axes[2].plot(step_list,var_dict_percent["tr2"],color=color[2] )
        # axes[2].plot(step_list[var_max_idxs["tr2"]],var_dict_percent["tr2"][var_max_idxs["tr2"]],marker="*",color="black",markersize=10)
        # axes[2].text(step_list[var_max_idxs["tr2"]], var_dict_percent["tr2"][var_max_idxs["tr2"]],"%.3f" % var_dict_percent["tr2"][var_max_idxs["tr2"]] + " %\n\n", fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        # axes[2].yaxis.set_major_locator(MaxNLocator(5))
        # axes[2].xaxis.set_major_locator(MaxNLocator(5))
        # axes[2].set_xlabel(mod_par.group(1)+" variation(rate)",fontsize=15)
        # axes[2].set_title("fall time (%)",fontsize=20)
        # axes[2].tick_params(axis='x', labelsize=15)
        # axes[2].tick_params(axis='y', labelsize=15)
        # # axes[2].set_ylim(min(data_dict["tr2"]),max(data_dict["tr2"]))
        # axes[2].yaxis.grid()
        # axes[2].set_ylim(y_low,y_high)

        plt.show()