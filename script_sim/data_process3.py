from logfile_process import logfile_reader
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from operator import itemgetter
from matplotlib.ticker import FormatStrFormatter
import numpy as np
def anlze_log_file(meas_params=[],aging_coffs=[],freq_coff=None,logfile="",tol_range=[0,None],freq_sel=None,ref_dict=None):
    # range 的設計是 為了縮小繪圖的區間範圍 若 step 是 100 simulation 0.01 per step 則 [0,50] 可以模擬出 1~1.5 倍的模擬結果 預設是全部print 出
    # freq_sel = None 時 與正常使用時一樣
    data = logfile_reader(logfile)
    df = None
    #定義要觀察的量測指標
    data_dict={}
    step_dict = {}
    ref = {}
    freq_idxs=[]
    if freq_sel is not None:
        freq_idxs = [i for i in range(len(data[freq_coff])) if data[freq_coff][i] == freq_sel]
    for aging_coff in aging_coffs:
        if freq_sel is not None:#若是log檔內包含多種頻率 需要進行資料處理 freq coff 有兩種type freq 或是 r_freq
            step_dict[aging_coff] =[float("%.2f"%x) for x in list(itemgetter(*freq_idxs)(data[aging_coff]))]
        else :
            step_dict[aging_coff] = [float("%.2f" % x) for x in data[aging_coff]]  # 因為在spice 的模擬過程中 1 step 為 0.01 精度故調整為0.2f
    ref_ll=[]
    for aging_coff in aging_coffs:
        ref_ll=ref_ll+[ i for i in range(len(step_dict[aging_coff])) if step_dict[aging_coff][i]==0]
    ref_idx=max(set(ref_ll), key=ref_ll.count)
    for meas_param in meas_params:
        if freq_sel is not None:
            data_dict[meas_param] = list(itemgetter(*freq_idxs)(data[meas_param]))[0:tol_range[1]]
            if ref_dict is not None and meas_param!="vth": # Vth 的 比較方式 不同於 indicator 是以inject =0 當作是最基礎狀態 屏除 freq 帶來的量測誤差
                #1. meas_param 設置為 None 的話 若ref_dict 設置為 None 則200 khz 會以 200khz inject =0 為基準 60khz 會以 60 khz inject =0 為基準
                #2. meas_param設置為 None 的話 若ref_dict 設置為 200 單頻得到的結果 則200 khz 會以 200khz inject =0 為基準 60khz 會以 200 khz inject =0 為基準
                #meas_param設置為 vth 的話 若ref_dict 設置為 200 單頻得到的結果  則200 khz 會以 200khz inject =0 為基準 60khz 會以 200 khz  inject =0 為基準 vth 則會走第一個選項
                ref[meas_param]=ref_dict[meas_param]
            else:
                ref[meas_param]=data_dict[meas_param][ref_idx]
        else:
            data_dict[meas_param]=data[meas_param][0:tol_range[1]]
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
        # if meas_param=="duty_ratio":#因為efficenicy 是本身就是用percentage 表示 所以採用鄉檢的方式去呈現下降的比例是多少
        #     data_dict[meas_param] = [float("%.4f" % (x)) for x in data_dict[meas_param]][tol_range[0]:]
        #
        #     var_dict_percent[meas_param]=[(x-ref[meas_param])*100for x in data_dict[meas_param]]
        #     var_dict[meas_param]=var_dict_percent[meas_param]
        # else :
        data_dict[meas_param] = [float("%.4f" % (x / ref[meas_param])) for x in data_dict[meas_param]][tol_range[0]:]  # 數舉處理後為比例 .4f 代表取到 百分位後第二位
        var_dict[meas_param] = [x - 1 for x in data_dict[meas_param]]  # 資料全部減1 獲取變化的"幅度" <0 為減少 >0 為增加
        var_dict_percent[meas_param] = [x * 100 for x in var_dict[meas_param]]

        var_max_dict[meas_param]=max(var_dict[meas_param],key=abs) #找出變化最大變化的"幅度"
        var_max_idxs[meas_param]=var_dict[meas_param].index(var_max_dict[meas_param])#    # 把變化最大的資料 的index 儲存起來

    for aging_coff in aging_coffs:
        step_dict[aging_coff] =step_dict[aging_coff][tol_range[0]:tol_range[1]]
    # 把變化最大的資料 print 出來
    # print(" rise time increase rate  : {} %".format( (  var_max_dict["tr1"]*100) )  )
    # print(" fall time increase rate  : {} %".format( ( var_max_dict["tr2"]*100) )  )
    # print(" maximum id current  increase rate : {} % ".format( ( var_max_dict["imax"]*100) )  )
    return df,data_dict,step_dict,var_dict,var_dict_percent,var_max_idxs,var_max_dict,ref

def circuit_data_process(device_name,
                         circuit_condition,R_freq_dict,
                         device_condition,
                         simulation_range=[0,None],
                         freq_sel_cir=[60000],
                         freq_sel_device=[60000],
                         ):
    meas_params = ["p_sw", "voutpp", "eff", "t_set", "duty_ratio", "vout_avg","i_drive"]
    aging_coffs = ["tol1"]
    title_ll = ["switching loss (%)", "ripple voltage (%)", "efficiency(%)", "set up time (%)",
                "switching on\nduty ratio (%)", "output voltage(%)", "Ids(%)" ]
    tol_range1 = simulation_range
    logfile1 = "./result/LTC1871/{}/{}/LTC1871-7_F09.log".format(device_name, circuit_condition)  # 轉為觀測用 的指標
    #### INITIAL 初始設定###############
    ###########################
    if len(freq_sel_cir)==1:
        # 今天若是做單頻率的測試 則 橫坐標的 Vth 以該頻率量測到的值去做比例分析
        # rj1p12bbd" 這顆device range 20000 範圍設置 [0 44] 之下 可以看到此系統崩潰前的一些前兆 set up time,ripple voltage(易量測) , switching loss , Ids eff,dutyratio
        # efficenicy 雖然在[0 44] 之下 稍微下降肇因為 switching loss ripple voltage 稍微 上升
        #
        pd_data, data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict,ref_out= anlze_log_file(
            meas_params=meas_params,
            aging_coffs=aging_coffs,
            logfile=logfile1,
            tol_range=tol_range1,
            freq_coff="r_freq",
            freq_sel=R_freq_dict[freq_sel_cir[0]],
            )
        # ########################將 step 的 param variation 轉為觀測用 的指標 VTO (%) -->threshold voltage (%)
        meas_params2 = ["tr1", "tr2", "imax", "vth"]
        aging_coffs2 = ["tol1"]
        param_sel = "vth"
        tol_range2 = [0, None]
        logfile2 = "./result/{}/{}/1/{}_MOS_N_VTO.log".format(device_name, device_condition, device_name)  # 轉為觀測用 的指標
        # freq_sel = freq_sel_device  # 若今天是模擬Circuit 是單一頻率 而 模擬元件是多重頻率 會需要這個敘述 因為上面會走 if 條件式 造成 _freq_sel=None 用到這裡
        pd_data, data_dict2, step_dict2, var_dict2, var_dict_percent2, var_max_idxs2, var_max_dict2,ref_out2= anlze_log_file(
            meas_params=meas_params2,
            aging_coffs=aging_coffs2,
            logfile=logfile2,
            tol_range=tol_range2,
            freq_coff="freq",
            freq_sel=freq_sel_device[0],
            )
        step_match_idx = [step_dict2['tol1'].index(i) for i in step_dict['tol1']]
        # ##########將 converter 電路中 模擬的 tol1 step 改在元件中模擬的 電路 tol1 step 找尋其IDX 以達到 parameter 如 Vth variation 的matching
        ######################################
        step_list = [var_dict_percent2[param_sel][i] for i in step_match_idx]
        # step_list = [x + 1 for x in step_dict[aging_coffs[0]]]
        # y_low = min([x for k in var_dict_percent.values() for x in k]) - 5
        # y_high = max([x for k in var_dict_percent.values() for x in k]) + 5

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
            axes[idx].set_xlim(min(step_list), max(step_list))  # 因欸變換到 Vth 變化幅度為 X 軸 故需要 社從0 開始 若為 step 為橫軸 則我們起點需要設為1
            axes[idx].yaxis.grid()
            axes[idx].set_ylim(y_low, y_high)
        fig.suptitle("Circuit Performance With Vth Aging Variation 0%% ~ %d%%"%max(step_list),fontsize=20,style="italic",weight="bold")
        plt.show()
    else:
        pass
        #

        pd_data10, data_dict10, step_dict10, var_dict10, var_dict_percent10, var_max_idxs10, var_max_dict10, ref_out10 = anlze_log_file(
            meas_params=meas_params,
            aging_coffs=aging_coffs,
            logfile=logfile1,
            tol_range=[0, None],
            freq_coff="r_freq",
            freq_sel=R_freq_dict[max(freq_sel_cir)]# 在分析頻率中選擇 最大的頻率當作 referance
        )
        data_collections_cir = list(map(lambda x:anlze_log_file(
            meas_params=meas_params,
            aging_coffs=aging_coffs,
            logfile=logfile1,tol_range=tol_range1,
            freq_coff="r_freq",
            freq_sel=R_freq_dict[x],
            ref_dict=ref_out10#174行需要去做調整 或不做調整 174 設None 代表 認定 inject param =0 為 brand new
            ),freq_sel_cir))
#         pd_data, data_dict, step_dict, var_dict, var_dict_percent, var_max_idxs, var_max_dict
        meas_params2 = ["tr1", "tr2", "imax", "vth"]
        aging_coffs2 = ["tol1"]
        param_sel = "vth"
        tol_range2 = [0, None]
        logfile2 = "./result/{}/{}/1/{}_MOS_N_VTO.log".format(device_name, device_condition, device_name)  # 轉為觀測用 的指標
        # freq_sel = freq_sel_device  # 若今天是模擬Circuit 是單一頻率 而 模擬元件是多重頻率 會需要這個敘述 因為上面會走 if 條件式 造成 _freq_sel=None 用到這裡
        pd_data20, data_dict20, step_dict20, var_dict20, var_dict_percent20, var_max_idxs20, var_max_dict20, ref_out20=anlze_log_file(
            meas_params=meas_params2,
            aging_coffs=aging_coffs2,
            logfile=logfile2,
            tol_range=tol_range2,
            freq_coff="freq",
            freq_sel=max(freq_sel_device)
        )
        data_collections_dev = list(map(lambda x:anlze_log_file(
            meas_params=meas_params2,
            aging_coffs=aging_coffs2,
            logfile=logfile2,
            tol_range=tol_range2,
            freq_coff="freq",
            ref_dict=None,#151行需要做調整
            freq_sel=x
            ),freq_sel_device))

        freq_sel_cir_k=[x/1E3 for x in freq_sel_cir]
        fig_1_graph_num = 3
        fig, axes = plt.subplots(1,fig_1_graph_num , figsize=(20, 10))
        fig2, axes2 = plt.subplots(1,len(meas_params)-fig_1_graph_num, figsize=(20, 10))

        rm_param_sel="vout_avg" # 因為幾乎在不同頻率下沒有麼差異 趨進 0 %  Vth point 繪圖 不予繪圖
        # meas_params_rm=[x for x in meas_params if x != rm_param_sel]
        fig3, axes3 = plt.subplots(1, fig_1_graph_num, figsize=(20, 10))
        fig4, axes4 = plt.subplots(1, len(meas_params) - fig_1_graph_num, figsize=(20, 10))

        fix_points = [10,20,30,40,45]
        sns.set(context='notebook', style='whitegrid')
        step_match_idx=[]
        step_list=[]
        drawing_range=[]
        # 總圖 包含了各個frequency 的變化
        # 可以知道 隨著老化狀況變的嚴重 各個指標的影響 呈現 Exponential 式的增長
        # 頻率越高者這個狀況越嚴重 對於老化帶來的影響會放大
        # 第一章圖包含了 switching loss  ripple voltage efficiency
        # switching loss 再高頻方面 隨著老化的增加 會呈現 exponential式 的增加
        for freq_idx,(data_collect_cir,data_collect_dev) in enumerate(zip(data_collections_cir,data_collections_dev)):
            pass
            pd_data1, data_dict1, step_dict1, var_dict1, var_dict_percent1, var_max_idxs1, var_max_dict1,ref_out1= data_collect_cir
            pd_data2, data_dict2, step_dict2, var_dict2, var_dict_percent2, var_max_idxs2, var_max_dict2,ref_out2= data_collect_dev
            # if freq_idx==0:#認為 VTO 增幅固定 則Vth得出來的值 會相同 若認定 threshold votage 與 freq 有關 把if 這段去掉即可
            step_match_idx = [step_dict2['tol1'].index(i) for i in step_dict1['tol1']]
            step_list = [var_dict_percent2[param_sel][i] for i in step_match_idx]
            drawing_range.append(max(step_list))
            for idx, (indicator, title) in enumerate(zip(meas_params[0:fig_1_graph_num], title_ll[0:fig_1_graph_num])):
                color = sns.color_palette("tab10")
                for fix_point in fix_points:
                    axes[idx].axvline(x=fix_point, color='0.7', linestyle='--')
                axes[idx].plot(step_list, var_dict_percent1[indicator], color=color[freq_idx],label=str(freq_sel_cir_k[freq_idx])+"kHz")
                axes[idx].yaxis.set_major_locator(MaxNLocator(5))
                # axes[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axes[idx].xaxis.set_major_locator(MaxNLocator(5))
                axes[idx].set_xlabel("Vth" + " variation(%)", fontsize=20)
                axes[idx].set_title(title, fontsize=20)
                axes[idx].tick_params(axis='x', labelsize=20)
                axes[idx].tick_params(axis='y', labelsize=20)
                axes[idx].set_xlim(0,min(drawing_range))
                # 限制了繪圖區域
                axes[idx].legend(loc='best', fontsize=15, title="Frequency", title_fontsize=15)
                # axes[idx].set_xticks(list(axes[idx].get_xticks()) + fix_points)# 防止 set xlim 失效
                axes[idx].yaxis.grid()
            fig.suptitle("Circuit Performance With Vth Aging Variation 0%% ~ %d%%" % min(drawing_range), fontsize=30,
                         style="normal", weight="bold")
            for idx, (indicator, title) in enumerate(zip(meas_params[fig_1_graph_num:], title_ll[fig_1_graph_num:])):
                color = sns.color_palette("tab10")
                for fix_point in fix_points:
                    axes2[idx].axvline(x=fix_point, color='0.7', linestyle='--')
                axes2[idx].plot(step_list, var_dict_percent1[indicator], color=color[freq_idx],label=str(freq_sel_cir_k[freq_idx])+"kHz")
                axes2[idx].yaxis.set_major_locator(MaxNLocator(4))
                # axes[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axes2[idx].xaxis.set_major_locator(MaxNLocator(4))
                axes2[idx].set_xlabel("Vth" + " variation(%)", fontsize=15)
                axes2[idx].set_title(title, fontsize=20)
                axes2[idx].tick_params(axis='x', labelsize=15,rotation=30)
                axes2[idx].tick_params(axis='y', labelsize=15,rotation=30)
                axes2[idx].set_xlim(0,min(drawing_range))
                axes2[idx].legend(loc='best', fontsize=15, title="Frequency", title_fontsize=15)
                # axes2[idx].set_xticks(list(axes2[idx].get_xticks()) + fix_points)# 防止 set xlim 失效
                axes2[idx].yaxis.grid()
                # handles, labels = axes2[idx].get_legend_handles_labels()
                # fig2.legend(handles, labels, loc='center left',title="Test Frequency",title_fontsize=15,fontsize=15)
            fig2.suptitle("Circuit Performance With Vth Aging Variation 0%% ~ %d%%" % min(drawing_range), fontsize=30,
                          style="normal", weight="bold")
        ## 固定 Vth point 繪圖
        for fix_idx,fix_point in enumerate(fix_points):
            sns.set(context='notebook', style='whitegrid')
            fix_indicator_y = {x: [] for x in meas_params}
            for freq_idx,(data_collect_cir,data_collect_dev) in enumerate(zip(data_collections_cir, data_collections_dev)):
                pass
                pd_data1, data_dict1, step_dict1, var_dict1, var_dict_percent1, var_max_idxs1, var_max_dict1, ref_out1 = data_collect_cir
                pd_data2, data_dict2, step_dict2, var_dict2, var_dict_percent2, var_max_idxs2, var_max_dict2, ref_out2 = data_collect_dev
                # if freq_idx==0:#認為 VTO 增幅固定 則Vth得出來的值 會相同 若認定 threshold votage 與 freq 有關 把if 這段去掉即可
                step_match_idx = [step_dict2['tol1'].index(i) for i in step_dict1['tol1']]
                step_list = [var_dict_percent2[param_sel][i] for i in step_match_idx]
                for indicator in meas_params:
                    fix_indicator_y[indicator].append(np.interp(fix_point, step_list, var_dict_percent1[indicator]))
            for idx, (indicator, title) in enumerate(zip(meas_params[0:fig_1_graph_num], title_ll[0:fig_1_graph_num])):
                color = sns.color_palette("tab10")
                var_percnt_ll = fix_indicator_y[indicator]
                var_percnt_ll = var_percnt_ll - var_percnt_ll[0]
                # var_percnt_ll = [round((x/var_percnt_ll[0]- 1) * 100, 2) for x in var_percnt_ll]  # 選擇哪一個頻率當作referance -1 為 freqll 最後一個頻率 0 為第一個
                axes3[idx].plot(freq_sel_cir_k, var_percnt_ll, color=color[fix_idx],label="%d %%"%fix_point)
                axes3[idx].yaxis.set_major_locator(MaxNLocator(4))
                # axes[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axes3[idx].xaxis.set_major_locator(MaxNLocator(4))
                axes3[idx].set_xlabel("Frequency", fontsize=20)
                axes3[idx].set_title(title, fontsize=20)
                axes3[idx].tick_params(axis='x', labelsize=20)
                axes3[idx].tick_params(axis='y', labelsize=20)
                axes3[idx].legend(loc='best', fontsize=15, title="Vth Aging Point", title_fontsize=15)
                axes3[idx].set_xticks(list(axes3[idx].get_xticks())+[min(freq_sel_cir_k)])
                axes3[idx].yaxis.grid()
            for idx, (indicator, title) in enumerate(zip(meas_params[fig_1_graph_num:], title_ll[fig_1_graph_num:])):
                color = sns.color_palette("tab10")
                var_percnt_ll = fix_indicator_y[indicator]
                var_percnt_ll = var_percnt_ll-var_percnt_ll[0]
                # var_percnt_ll = [round((x / var_percnt_ll[0] - 1) * 100, 2) for x in var_percnt_ll]  # 選擇哪一個頻率當作referance -1 為 freqll 最後一個頻率 0 為第一個
                axes4[idx].plot(freq_sel_cir_k, var_percnt_ll, color=color[fix_idx],label="%d %%" % fix_point)
                axes4[idx].yaxis.set_major_locator(MaxNLocator(5))
                # axes4[idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                axes4[idx].xaxis.set_major_locator(MaxNLocator(5))
                axes4[idx].set_xlabel("Frequency", fontsize=15)
                axes4[idx].set_title(title, fontsize=20)
                axes4[idx].tick_params(axis='x', labelsize=15, rotation=30)
                axes4[idx].tick_params(axis='y', labelsize=15, rotation=30)
                axes4[idx].legend(loc='best', fontsize=15, title="Vth Aging Point", title_fontsize=15)
                axes4[idx].set_xticks(list(axes4[idx].get_xticks())+[min(freq_sel_cir_k)])
                axes4[idx].yaxis.grid()
                # handles, labels = axes2[idx].get_legend_handles_labels()
                # fig2.legend(handles, labels, loc='center left',title="Test Frequency",title_fontsize=15,fontsize=15)
        fig4.suptitle("Frequency With Vth Aging Variation {}% ~ {}%".format(min(fix_points),max(fix_points)), fontsize=30,
                          style="normal", weight="bold")
        fig3.suptitle("Frequency With Vth Aging Variation {}% ~ {}%".format(min(fix_points),max(fix_points)), fontsize=30,
                      style="normal", weight="bold")

        plt.show()

