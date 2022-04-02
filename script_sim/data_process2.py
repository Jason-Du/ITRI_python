from PyLTSpice.LTSteps import LTSpiceLogReader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os import listdir
import re
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from data_process3 import circuit_data_process,anlze_log_file
DRAW_1=True
DRAW_2=False
MAX_VARIATION_ANALY=False
CIRCUIT_ANALY=False
CIRCUIT_Single=False#與# CIRCUIT_ANALY 是綁定一起的 Single 指的是頻率是single 的
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

# "rj1p12bbd"
# "rq3p300bh"
device_name="rj1p12bbd"#DRAW_1 MAX_VARIATION_ANALY CIRCUIT_ANALY 皆會用到
device_condition="all_freq_7.5Vg_40Vds_70A" #全部都會用到
circuit_condition="all_freq"  #CIRCUIT_ANALY 會用到
# "all_freq"
# "Rk_60kHz"
#"40khz_7.5Vg_40Vds_1.6R"
# all_freq_7.5Vg_40Vds_2.5R
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
if __name__ == '__main__':
    if CIRCUIT_ANALY:
        # 在分析的過程中 若今天"rcj510n25" device 取 VTO 為影響 threshold voltage 的主要因素 下去模擬 25step 0.01 rate/step
        # switching loss 在 第7 次模擬結果後 升的超級高(第 10 step) 原因是因為站空比拉大 (90%) P_Sw 的 spice 的 script 裡面是 V*I 沒有足夠的時間下降 導製 ZVS ZCS 失去效用
        # switching loss (第 10 step) 之後開始劇降 原因是因為 output voltage 開始降低 拉不到 規格中的 42 V
        # 會發現系統後半段 的模擬 efficiency掉得很快 是因為 電壓已經拉不上去 拉不到 規格中的 42 V
        # 原因是因為沒掉穩太狀態 所以能參考的資料 在模擬時間有限的狀況下(10ms) 僅能參考 7step 的資料資訊
        # set time (第 10 step) 之前 呈現 EXPONENTIAL GRAPH 式 的增長 driving 的能力變弱了 上升期 階段的流經電流慢慢隨著老化在下降
        # 上述原因導致了 需要比較長的時間來做 set up 進一步導致 diode 損壞 I peak avg break
        # duty ratio 在前面7 step 模擬的 variation 算是可以接受
        # ripple voltage 不適合拿來做為 一個 因為太過浮動 但當系統崩快時 output voltage 開始降低 拉不到 規格中的 42 V 也算是崩快的一種依據
        # 指標警示 可選擇 set up time 指標警示系統 可訂 530%
        # Switching 作為 指標警示系統 可訂 6.37%
        # efficiency 在前面9個step 變化不明顯 是因為 系統利用 條大佔空比 去彌補 trade -off switching loss 增加
        # 容易被 user 量測的指標 : 電壓 有包含 ripple voltage set up time switching on duty ratio output voltage
        freq_ll = [60000, 80000, 100000, 150000, 200000]
        circuit_data_process(device_name=device_name,
                             circuit_condition=circuit_condition,
                             R_freq_dict=R_freq_dict,
                             device_condition=device_condition,
                             simulation_range=[0, 44],
                             freq_sel_cir =freq_ll,#若為單頻率 選擇 [60000] 若無頻率選擇 [None]
                             freq_sel_device =freq_ll,#若為單頻率 選擇 [60000] 若無頻率選擇 [None]
                             )
        # RQ3P300BH [20,38]
        # RJ1P12BBD [0,44]
    if MAX_VARIATION_ANALY:
        files = listdir("./result/{}/{}/{}".format(device_name,device_condition,param_num))
        meas_params = ["tr1","tr2","imax","vth"]
        aging_coffs = ["tol1"]
        var_max_idxs={}
        one_max_variation={}
        max_variations={"measure_value":[],"measure_name":[],"param_name":[]}
        param_sel="tr2"

        for f_idx, f in enumerate(files):
            pass
            logfile = "./result/{}/{}/{}/".format(device_name,device_condition,param_num)+f
            pd_data,data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict,ref_out= anlze_log_file(meas_params=meas_params,
                                                                                                     aging_coffs=aging_coffs,
                                                                                                     logfile=logfile,
                                                                                                     tol_range=[0,35],
                                                                                                     freq_coff="freq",
                                                                                                     freq_sel=60000
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
        pd_data,data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict,ref_out= anlze_log_file(meas_params=meas_params,
                                                                                                      aging_coffs=aging_coffs,
                                                                                                      logfile=logfile,
                                                                                                      tol_range=[0,None]
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
        logfile = "./result/{}/{}/1/{}_MOS_N_VTO.log".format(device_name,device_condition,device_name)
        pattern = re.compile(r"{}_(.*).log".format(device_name))
        mod_par = re.search(pattern,logfile)
        meas_params = ["tr1", "tr2", "imax","vth"]
        title_ll=["rise time (%)","fall time (%)","Ids (%)","Vth(%)"]
        aging_coffs = ["tol1"]
        freq_ll=[60000,80000,100000,150000,200000]#若為單一頻率架購則設為None
        # freq_ll=[200000]
        if len(freq_ll)==1:
            pd_data,data_dict, step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict,ref_out=anlze_log_file(meas_params=meas_params,
                                                                                                    aging_coffs=aging_coffs,
                                                                                                    logfile=logfile,
                                                                                                    tol_range=[0,None],
                                                                                                    freq_coff="freq",
                                                                                                    freq_sel=freq_ll[0]
                                                                                                    )
            step_list=[x+1for x in step_dict[aging_coffs[0]]]
            #全部評比使用同一個座標軸
            # y_low=min([x for k in var_dict_percent.values() for x in k ])-5
            # y_high=max([x for k in var_dict_percent.values() for x in k ])+5
            fig,axes=plt.subplots(1,len(meas_params),figsize=(20, 10))
            # #繪製模擬時間內的最大電流 Rise time fall time
            color = sns.color_palette("Set1")
            for idx,(indicator,title) in enumerate(zip(meas_params,title_ll)):
                y_low=min(var_dict_percent[indicator])-5
                y_high=max(var_dict_percent[indicator])+5
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
                axes[idx].set_xlim(min(step_list),max(step_list))
                axes[idx].yaxis.grid()
                axes[idx].set_ylim(y_low,y_high)
            plt.show()
        else:
            pd_data0, data_dict0, step_dict0, var_dict0, var_dict_percent0, var_max_idxs0, var_max_dict0, ref_out0= anlze_log_file(
                meas_params=meas_params,
                aging_coffs=aging_coffs,
                logfile=logfile,
                tol_range=[0, None],
                freq_coff="freq",
                freq_sel=min(freq_ll)# 在分析頻率中選擇 最大的頻率當作 referance
                )

            data_collections = list(map( lambda x:anlze_log_file(
                meas_params=meas_params,
                aging_coffs=aging_coffs,
                logfile=logfile,
                tol_range=[0, None],
                freq_coff="freq",
                ref_dict=ref_out0,# 若要以上面的頻率當作全體頻率的referance 則 設為 ref_out0
                freq_sel=x
                ),freq_ll))
            # print(len(data_collections[0]))
            # print(len(list(zip(*data_collections))))
            ##################Inject param (%) X 軸 Y 軸 為各個指標 #####################
            # 從此圖發現 在 inject range 固定狀況下 頻率越高者帶來的變化幅度較小
            # Vth 理解為 VGS 突破的電壓障礙 開始產生電流 高頻下的等效容值(Ciss)會下降 相對來說 vth 是下降的
            # 但我們inject aging param 的 目的在於說 要讓threshold votage 偏移 觀察這個偏移的影響下 對系統以及對 Device Failure indicator 的影響
            # 所以X軸 應改為 Vth 變化 來觀測
            #高頻下 Ciss 阻抗小 升壓快(上升速度) 在量測相同的電流之下 他會量到比較高的電壓
            fig, axes = plt.subplots(1, len(meas_params), figsize=(20, 10))
            sns.set(context='notebook', style='whitegrid')
            for freq_idx, data_collect in enumerate(data_collections):
                pd_data, data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict,ref_out=data_collect
                step_list = [x + 1 for x in step_dict[aging_coffs[0]]]
                color = sns.color_palette("Set1")
                for idx, (indicator, title) in enumerate(zip(meas_params, title_ll)):
                    y_low = min(var_dict_percent[indicator]) - 5
                    y_high = max(var_dict_percent[indicator]) + 5
                    axes[idx].plot(step_list, var_dict_percent[indicator], color=color[freq_idx],label=str(freq_ll[freq_idx]/1E3)+"kHz")
                    axes[idx].yaxis.set_major_locator(MaxNLocator(5))
                    axes[idx].xaxis.set_major_locator(MaxNLocator(5))
                    axes[idx].set_xlabel(mod_par.group(1) + " variation(rate)", fontsize=15)
                    axes[idx].set_title(title, fontsize=20)
                    axes[idx].tick_params(axis='x', labelsize=15)
                    axes[idx].tick_params(axis='y', labelsize=15)
                    # axes[idx].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
                    axes[idx].set_xlim(min(step_list), max(step_list))
                    axes[idx].legend(loc="best",fontsize=10, title="Test Frequency", title_fontsize=10)
                    axes[idx].yaxis.grid()
                    # axes[idx].set_ylim(y_low, y_high)
                    # handles, labels = axes[idx].get_legend_handles_labels()
                    # fig.legend(handles, labels, loc='upper center',title="Test Frequency",title_fontsize=15,ncol=len(freq_ll))
            # plt.show()
            ##################Vth(%) X 軸 Y 軸 為 個個指標 ##############################
            # 但因為 Vth 是我們觀測老化的一個重要指標 所以 我們將 Vth 轉為 X 軸
            # 在相同的老化狀況下 高頻帶來對 Power MOSFET 的risetime falltime 影響 相對低頻來說影響較沒那麼大
            # 原因是因為 高頻下的 店如阻抗較小 Gate drive 相對來說有比較強的能力去 drive 這柯mosfet  rise time fall performance 比較好
            rm_param_sel="vth"#選擇作為x軸的 指標
            fix_point = 45  # 固定要觀看的X值 在這裡為Vth 50% driftstep_list
            step_list =[]
            meas_params_rm = [x for x in meas_params if x != rm_param_sel]
            fig3, axes3 = plt.subplots(1, len(meas_params_rm), figsize=(20, 10))
            sns.set(context='notebook', style='whitegrid')
            color = sns.color_palette("Set1")
            for freq_idx, data_collect in enumerate(data_collections):
                pd_data, data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict,ref_out=data_collect
                # step_list = [x + 1 for x in step_dict[aging_coffs[0]]]
                # if freq_idx==0:#認為 VTO 增幅固定 則Vth得出來的值 會相同 若認定 threshold votage 與 freq 有關 把if 這段去掉即可
                step_list = var_dict_percent["vth"]
                for idx, (indicator, title) in enumerate(zip(meas_params_rm, title_ll)):
                    axes3[idx].plot(step_list, var_dict_percent[indicator], color=color[freq_idx],label=str(freq_ll[freq_idx]/1E3)+"kHz")
                    axes3[idx].axvline(x=fix_point, color='k', linestyle='--')
                    axes3[idx].yaxis.set_major_locator(MaxNLocator(5))
                    axes3[idx].xaxis.set_major_locator(MaxNLocator(5))
                    axes3[idx].set_xlabel("Threshold Voltage variation(%)", fontsize=15)
                    axes3[idx].set_title(title, fontsize=20)
                    axes3[idx].tick_params(axis='x', labelsize=15)
                    axes3[idx].tick_params(axis='y', labelsize=15)
                    axes3[idx].set_xticks(list(axes3[idx].get_xticks()) + [fix_point])
                    # axes[idx].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
                    # axes[idx].set_xlim(0, max(step_list))
                    # axes[idx].legend(fontsize=10, title="Test Frequency", title_fontsize=15)
                    axes3[idx].legend(loc='best',fontsize=10,title="Test Frequency",title_fontsize=10)
                    axes3[idx].xaxis.grid()
                    # axes[idx].set_ylim(y_low, y_high)
                    # handles, labels = axes3[freq_idx].get_legend_handles_labels()
                    # fig3.legend(handles, labels, loc='upper center',title="Test Frequency",title_fontsize=15,ncol=len(freq_ll))

            #################Frequency X 軸 Y 軸 取變化量最大者##########################
            #  將Vth 作為X軸 固定可以發現 頻率越高者 帶來的變化越大
            # 頻率越高者在相同老化情形下對系統的影響可能較大
            fix_indicator_y= {x:[] for x in meas_params}
            for freq_idx, data_collect in enumerate(data_collections):
                pd_data, data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict,ref_out= data_collect
                # if freq_idx == 0:#認為 VTO 增幅固定 則Vth得出來的值 會相同 若認定 threshold votage 與 freq 有關 把if 這段去掉即可
                step_list = var_dict_percent["vth"]
                for indicator in meas_params_rm:
                    fix_indicator_y[indicator].append(np.interp(fix_point,step_list,var_dict_percent[indicator]))
            fig2, axes2 = plt.subplots(1, len(meas_params_rm), figsize=(20, 10))
            color = sns.color_palette("Set1")
            freq_ll_k = [int(x / 1E3) for x in freq_ll]
            for idx, (indicator, title) in enumerate(zip(meas_params_rm, title_ll)):
                var_percnt_ll = fix_indicator_y[indicator]
                var_percnt_ll = [round((x / var_percnt_ll[-1] - 1) * 100, 2) for x in var_percnt_ll]# 選擇哪一個頻率當作referance -1 為 freqll 最後一個頻率 0 為第一個
                axes2[idx].plot(freq_ll_k, var_percnt_ll, color=color[idx])
                axes2[idx].yaxis.set_major_locator(MaxNLocator(5))
                axes2[idx].set_xticks([x for x in range (min(freq_ll_k),max(freq_ll_k)+20,20)])
                axes2[idx].set_xlabel("Frequency (kHz)", fontsize=15)
                axes2[idx].set_title(title, fontsize=20)
                axes2[idx].tick_params(axis='x', labelsize=15)
                axes2[idx].tick_params(axis='y', labelsize=15)

                axes2[idx].xaxis.grid()
            plt.show()