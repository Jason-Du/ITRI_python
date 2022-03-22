import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator as op
import statistics
import copy
from statistics import mean
from itertools import cycle, islice
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import scipy.io as matloader
temp_ll=[40.0,100.0,150.0,200.0,210.0,220.0,230.0,240.0,250.0]

def aging_AF_coff(A,B,C,Temp,Vgate,a):
    pass
    AF_coff=A*(10**(C*Temp))*(10**(B*Vgate+a))
    return AF_coff
# 藉由 aging 溫度 Gate 端偏壓 去計算出AF cofficient
# 範例中圍論文提到藉由 IRF4XX 系列 實驗數據結果可以得到的 A,B,C 的參數值
def temp_dr_analyze(read_file,temp_ll):
    all_temp=[]
    df=pd.read_csv(read_file)
    ####################
    # for temp_data in df.keys()[1::2]:
    #     for temp_inform in set(df[temp_data].drop()):
    #         all_temp.append(temp_inform)
    # temp_duration_dict={s_key:[] for s_key in sorted(set(all_temp))}
    #可得 最大temp 的集合
    #############################
    temp_duration_dict = {s_key: [] for s_key in sorted(set(temp_ll))}
    temp_duration_dict["run"] = []
    for idx,(time_data,temp_data) in enumerate(zip(df.keys()[0::2],df.keys()[1::2])):
        temp_duration_dict["run"].append("run {}".format(idx + 1))
        for temp_inform in set(df[temp_data].dropna()):
            require_index=df.index[ df[temp_data] == temp_inform].tolist()
            temp_duration_dict[temp_inform].append(df[time_data][require_index[-1]]-df[time_data][require_index[0]])
        for i in set(temp_duration_dict.keys()).difference(set(df[temp_data].dropna())):
            if i !="run":
                temp_duration_dict[i].append(0)
    new_d = {str(key): value for key, value in temp_duration_dict.items()}
    df2=pd.DataFrame(new_d)
    return df2
def draw_temp_time(file_lib,MOSFET_num_ll):
    pass
    if len(MOSFET_num_ll)==1:
        print("Warning there is no 4 data please do not use this function")
    fig, axes = plt.subplots(2,2, figsize=(18, 10))
    for idx,MOSFET_num in enumerate(MOSFET_num_ll):
        df2 = temp_dr_analyze(file_lib+"/%s.csv"%MOSFET_num,temp_ll=temp_ll)
        color = sns.color_palette("rocket_r", as_cmap=False, n_colors=len(temp_ll))
        # temp_color_dict = list(islice(cycle(color), None,len(temp_ll)))
        sns.set(style='white')
        df2=df2.set_index("run")

        df2.plot(kind='barh', stacked=True, color=color,ax=axes[idx//2,idx%2])
        axes[idx//2,idx%2].tick_params(axis='y', labelsize=15,rotation=30)
        axes[idx//2,idx%2].tick_params(axis='x', labelsize=15,rotation=30)
        axes[idx//2,idx%2].yaxis.grid()
        # axes[idx//2,idx%2].legend(["{} $^\circ$C".format(i) for i in df2.columns],fontsize=15,title="Temperatutre",fancybox=True)
        axes[idx // 2, idx % 2].get_legend().remove()
        # axes[idx//2,idx%2].set_title("Power MOSFET %s High Temperature Test " %MOSFET_num, fontsize=10)
        axes[idx // 2, idx % 2].set_ylabel("", fontsize=10)
        axes[idx//2,idx%2].set_xlabel("%s Test Run Duration (min)"%MOSFET_num,fontsize=15)
        handles, labels = axes[idx//2,idx%2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left',fontsize=10,title="Temperatutre\n($^\circ$C)",fancybox=True)
    fig.suptitle("%d Power MOSFET High Temperature Test " %len(MOSFET_num_ll),fontsize=20)
def Calculate_MTTF(hours):
    return hours/(24*365)
def execution(Temp_dR=True,
    AF_COFF=False,
    COX_GATE=False,
    Vth_AGING=False,
    dR_AGING=False,
    Temp_time=False,
    Nominal_resistanc=0.25,
    MOSFET_rdson_ll = ['MOSFET 14', 'MOSFET 8', 'MOSFET 9', 'MOSFET 36', 'MOSFET 11', 'MOSFET 12'],
    MOSFET_TEMP_ll = ["dr14", "dr8", "dr9", "dr36", "dr11", "dr12"],
    MOSFET_SEL='MOSFET 36',
    temp_ll=temp_ll):
    if AF_COFF:
        Temp_points=100
        Vgate_point=5
        Temp_array=np.linspace(50,300,Temp_points)
        Vgate_array=np.linspace(4,24,Vgate_point)
        Temp_array=np.tile(Temp_array,(Vgate_point,1))
        Vgate_array=np.transpose(np.tile(Vgate_array,(Temp_points,1)))
        aging_AF_array=aging_AF_coff(A=-17.2,B=0.4,C=-0.15,Temp=Temp_array,Vgate=Vgate_array,a=0)
        #繪圖區塊
        fig,axes=plt.subplots(1,1,figsize=(20, 10))
        color = sns.color_palette("Set1")
        for idx,aging_AF in enumerate(aging_AF_array):
            axes.plot(Temp_array[0],aging_AF,color=color[idx],label="Vgate= {} V".format(Vgate_array[idx][0]) )
        plt.legend()
        plt.show()

    if COX_GATE:
        # 小結: 100 kHZ / 1MHZ 以 1kHz 當作 referance 來看 震盪幅度落在-0.5~4.8%  2.32%~-0.5%
        # 頻率越高者 其 variation 的範圍也就越大
        df = pd.read_csv('./paper_dataset/Cox_gate.csv')
        labels = ['1KHZ', '100KHZ', '1MHZ']
        X_list=['1KHZ_X','100KHZ_X','1MHZ_X']
        Y_list = ['1KHZ_Y', '100KHZ_Y', '1MHZ_Y']
        Compare_array=[]
        Vgate_array=[V_gate_sel for V_gate_sel in range(-8,7,1)]# 決定固定 Vgate 電壓下 操作在 1kHz 100kHz 1MHZ 狀況下對 Cox 造成的 variation
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        color = sns.color_palette("Set1")
        for idx,(X,Y) in enumerate(zip(X_list,Y_list)):
            axes[0].plot(df[X],df[Y], color=color[idx], label=labels[idx])# 繪製所有曲線
            Compare_array.append(np.interp(Vgate_array,df[X],df[Y]))

        Compare_array=np.array(Compare_array)
        Compare_array=np.transpose(Compare_array)
        Compare_array=np.array([((set1/set1[0])-1)*100 for set1 in Compare_array]) # 以1HZ 為referance

        axes[0].set_title("Cox(pf) ", fontsize=20)
        axes[0].set_xlabel("V(gate)", fontsize=15)
        axes[0].tick_params(axis='x', labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)
        axes[0].yaxis.grid()
        axes[0].legend()
         # 繪製 在相同V(gate) 以 1 kHZ 為ref 看 '100KHZ', '1MHZ' 與之的差異比較
        color = sns.color_palette("Set2")
        for idx in range(1,3):
            axes[1].plot(Vgate_array,Compare_array[:, idx], color=color[idx], label=labels[idx])  # 繪製100kHZ
            axes[1].plot(Vgate_array[Compare_array[:, idx].argmax()], Compare_array[:, idx][Compare_array[:, idx].argmax()],marker="*", color="black", markersize=10)
            axes[1].plot(Vgate_array[Compare_array[:, idx].argmin()], Compare_array[:, idx][Compare_array[:, idx].argmin()],marker="*", color="black", markersize=10)
            axes[1].text(Vgate_array[Compare_array[:, idx].argmax()], Compare_array[:, idx][Compare_array[:, idx].argmax()],"%.3f" % Compare_array[:, idx][Compare_array[:, idx].argmax()] + " %" + "\n" * idx, fontsize=12,color="k", style="italic", weight="bold", verticalalignment='center', horizontalalignment='right',rotation=0)
            axes[1].text(Vgate_array[Compare_array[:, idx].argmin()], Compare_array[:, idx][Compare_array[:, idx].argmin()],"%.3f" % Compare_array[:, idx][Compare_array[:, idx].argmin()]  + " %"+"\n\n"*idx, fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        axes[1].yaxis.grid()
        axes[1].set_title("Cox variation(%) ", fontsize=20)
        axes[1].set_xlabel("V(gate)", fontsize=15)
        axes[1].tick_params(axis='x', labelsize=15)
        axes[1].tick_params(axis='y', labelsize=15)
        axes[1].legend()
        plt.show()
    if Vth_AGING:
        # 小結 溫度造成的影響 非常大 可能使影響幅度 到達兩倍之差
        # 老化結果顯示 在純吋高壓狀態下 Aging 幅度 落在 20% 左右 高壓高溫下 則會到達50% 以上的variation
        df = pd.read_csv("./paper_dataset/Vth_aging.csv")
        df2 = pd.read_csv("./paper_dataset/Vth_aging_350K.csv")
        labels = ['AgingMOSFET 0', 'AgingMOSFET 2','AgingMOSFET 5', 'AgingMOSFET 6','AgingMOSFET 7']
        X_list=['AgingMOSFET0X', 'AgingMOSFET2X','AgingMOSFET5X','A   gingMOSFET6X','AgingMOSFET7X']
        Y_list = ['AgingMOSFET0Y', 'AgingMOSFET2Y', 'AgingMOSFET5Y', 'AgingMOSFET6Y','AgingMOSFET7Y']
        hour_array=[i for i in range(0,8,1)]
        hour_array.append(7.9)
        Compare_300k=[]
        Compare_350k=[]

        fig, axes = plt.subplots(1, 3, figsize=(20, 10))
        color=sns.color_palette("husl", 9)
        for idx,(X,Y) in enumerate(zip(X_list,Y_list)):
            axes[0].plot(df[X],df[Y], color=color[idx], label=labels[idx])# 繪製所有曲線
            axes[1].plot(df2[X], df2[Y], color=color[idx], label=labels[idx])  # 繪製所有曲線
            Compare_300k.append(np.interp(hour_array, df[X], df[Y]))
            Compare_350k.append(np.interp(hour_array, df2[X], df2[Y]))

        Compare_300k = np.array(Compare_300k)
        Compare_300k = np.array([((set1 / set1[0]) - 1) * 100 for set1 in Compare_300k])  # 以第0小時 為referance
        Compare_350k = np.array(Compare_350k)
        Compare_350k = np.array([((set1 / set1[0]) - 1) * 100 for set1 in Compare_350k])

        Compare_300k_avg=np.average(Compare_300k, axis=0) # 將每個小時的MOSFET 測試資料取其平均
        Compare_350k_avg = np.average(Compare_350k, axis=0)

        axes[2].plot(hour_array,Compare_300k_avg, color=color[0], label="300K Average Vth variation")
        axes[2].plot(hour_array,Compare_350k_avg, color=color[1], label="350K Average Vth variation")
        axes[2].plot(hour_array[Compare_300k_avg.argmax()], np.max(Compare_300k_avg),marker="*", color="black", markersize=10)
        axes[2].plot(hour_array[Compare_350k_avg.argmax()], np.max(Compare_350k_avg),marker="*", color="black", markersize=10)
        axes[2].text(hour_array[Compare_300k_avg.argmax()], np.max(Compare_300k_avg),"%.3f" % np.max(Compare_300k_avg) + " %" + "\n" , fontsize=12,color="k", style="italic", weight="bold", verticalalignment='center', horizontalalignment='right',rotation=0)
        axes[2].text(hour_array[Compare_350k_avg.argmax()], np.max(Compare_350k_avg),"%.3f" % np.max(Compare_350k_avg) + " %" + "\n", fontsize=12, color="k", style="italic", weight="bold",verticalalignment='center', horizontalalignment='right', rotation=0)



        axes[0].set_title("MOSFET Eletrical Stress T=300K Vgs=34V", fontsize=20)
        axes[0].set_xlabel("Aging Time (hours)", fontsize=15)
        axes[0].set_ylabel("Threshold voltage(V)", fontsize=15)
        axes[0].tick_params(axis='x', labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)
        axes[0].legend()
        axes[0].yaxis.grid()

        axes[1].set_title("MOSFET Eletrical Stress T=350K Vgs=34V", fontsize=20)
        axes[1].set_xlabel("Aging Time (hours)", fontsize=15)
        axes[1].set_ylabel("Threshold voltage(V)", fontsize=15)
        axes[1].tick_params(axis='x', labelsize=15)
        axes[1].tick_params(axis='y', labelsize=15)
        axes[1].legend()
        axes[1].yaxis.grid()

        axes[2].set_title("MOSFET Eletrical Stress Variation", fontsize=20)
        axes[2].set_xlabel("Aging Time (hours)", fontsize=15)
        axes[2].set_ylabel("Threshold voltage(%)", fontsize=15)
        axes[2].tick_params(axis='x', labelsize=15)
        axes[2].tick_params(axis='y', labelsize=15)
        axes[2].legend()
        axes[2].yaxis.grid()
        plt.show()

    if dR_AGING:
        # 呈現出 MOSFET 體質的不同
        # 為7次thermal cycling 觀察其 aging 的結果
        # MOSFET 最大增加幅度 可以到達 0.9% per 10 miunites
        # 在總共的 Aging 時間裡面 平均值落在 0.01% ~
        #可是為200度高溫下的老化情形 可以說明個體上的差異蠻大的 也可以當作是simulator inject veriation 的幅度
        # 可以在繪製一張圖表是關於 Aging 總時長的前半 後半 差異
        df_rdson = pd.read_csv("./paper_dataset/dr_aging_800min.csv")
        dict={}
        # labels = ['MOSFET 14', 'MOSFET 8','MOSFET 9', 'MOSFET 36','MOSFET 11','MOSFET 12']

        # X_list=['AgingMOSFET14X', 'AgingMOSFET8X' , 'AgingMOSFET9X','AgingMOSFET36X','AgingMOSFET11X','AgingMOSFET12X']
        # Y_list = ['AgingMOSFET14Y', 'AgingMOSFET8Y', 'AgingMOSFET9Y', 'AgingMOSFET36Y','AgingMOSFET11Y','AgingMOSFET12Y']
        sample_scale=30 # 單位為 min
        slope=[]

        color=sns.color_palette("tab10")
        fig2, axes2 = plt.subplots(1, 1, figsize=(20, 10))
        for idx,(X,Y,label) in enumerate(zip(df_rdson.columns[0::2],df_rdson.columns[1::2],MOSFET_rdson_ll)):
            df_rdson[Y][0] = 0
            df_rdson[Y]=(df_rdson[Y]/Nominal_resistanc)*100
            dict[label]=[(np.interp(idx+sample_scale-1,df_rdson[X],df_rdson[Y])-np.interp(idx,df_rdson[X],df_rdson[Y]))/sample_scale
                         for idx in range(0,int(df_rdson[X][df_rdson[X].last_valid_index()]-sample_scale+1),sample_scale)]# range(1,2,3) window size=3
            axes2.plot(df_rdson[X], df_rdson[Y], color=color[idx], label=MOSFET_rdson_ll[idx])  # 繪製所有曲線

        Aging_time_scales=[35,35,35,35,180,240,240]
        Temperatures=[250,240,230,220,210,210,210]
        initial_time=0
        Aging_time_tick=[]

        for idx,(age_time,Temperature) in enumerate(zip(Aging_time_scales,Temperatures)):
            axes2.text(initial_time+(age_time/2), 160, "{}$^\circ$C".format(Temperature), horizontalalignment='center', size=15, color="r",style="italic", weight="bold")
            initial_time = age_time + initial_time
            if(idx!=6):# 6等分 只需畫好 五條線
                axes2.axvline(x=initial_time,color='r', linestyle='--')


        axes2.set_title("Power MOSFET \u0394R Variation T=210~250$^\circ$C Vgs=10V Freq=1kHz VDS=4V ", fontsize=20)
        axes2.set_xlabel("Aging Time (min)", fontsize=15)
        axes2.set_ylabel("\u0394R variation(%)", fontsize=15)
        axes2.tick_params(axis='x', labelsize=15)
        axes2.tick_params(axis='y', labelsize=15)
        axes2.legend()
        axes2.yaxis.grid()


        sorted_keys, sorted_vals = zip(*sorted(dict.items(), key=op.itemgetter(1)))
        median_array = [np.round(statistics.median(i),3)for i in sorted_vals]
        max_array = [np.round(max(i), 3) for i in sorted_vals]
        min_array = [np.round(min(i), 3) for i in sorted_vals]

        fig, axes = plt.subplots(1, 1, figsize=(20, 10))
        color=sns.color_palette("Set2")
        sns.set(context='notebook', style='whitegrid')
        box_plot=sns.boxplot(data=sorted_vals, width=.18, palette=color,showfliers = False)
        sns.stripplot(data=sorted_vals, size=6, edgecolor="black", linewidth=.5,palette=color,jitter=0.05)

        axes.set_title("MOSFET Rdson Aging Rate Distribution", fontsize=20)
        axes.set_xticklabels(sorted_keys, size=15)
        axes.set_yticklabels(np.round(axes.get_yticks(),2),size=15)
        axes.set_ylabel("Aging Rate (%/{}min)".format(sample_scale), fontsize=15, rotation=90)
        axes.yaxis.grid()

        for label,median_value,max_value,min_value in zip(axes.get_xticks(),median_array,max_array,min_array):
            box_plot.text(label+0.25,median_value*1.005,"{}%".format(median_value),horizontalalignment='center',size=12,color="k",style="italic", weight="bold")
            box_plot.text(label + 0.25, max_value * 1.005, "{}%".format(max_value), horizontalalignment='center',size=12, color="k", style="italic", weight="bold")
            box_plot.text(label + 0.25, min_value * 1.005, "{}%".format(min_value), horizontalalignment='center',size=12, color="k", style="italic", weight="bold")
        plt.show()

    if Temp_time:
        #36 8 11 14
        # 呈現 mosfet 加熱的溫度分布 以及時長分布
        # 呈現出 mosfet 的加溫 36最為特別 是持續 250K 高溫持續烘烤
        Single=False
        if Single:
            df2=temp_dr_analyze(read_file="./paper_dataset/NASA_MOSFET_TEMP/dr8.csv",temp_ll=temp_ll)
            fig, axes = plt.subplots(1, 1, figsize=(20, 10))
            color = sns.color_palette("rocket_r", as_cmap=False, n_colors=len(temp_ll))
            temp_color_dict = list(islice(cycle(color), None,len(temp_ll)))

            sns.set(style='white')
            df2=df2.set_index("run")
            df2.plot(kind='barh', stacked=True, color=color,ax=axes)
            axes.tick_params(axis='y', labelsize=15)
            axes.tick_params(axis='x', labelsize=15)
            axes.yaxis.grid()
            # axes.legend(["{} $^\circ$C".format(i) for i in df2.columns],fontsize=15,title="Temperatutre",fancybox=True)
            axes.set_title("Power MOSFET High Temperature Test ", fontsize=20)
            axes.set_ylabel("Test Run No.", fontsize=15)
            axes.set_xlabel("Test Run Duration (min)",fontsize=15)
            plt.legend(bbox_to_anchor=(1.0, 1.0),fontsize=15, title="Temperatutre\n($^\circ$C)",fancybox=True)
            plt.show()
        else:
            draw_temp_time(file_lib="./paper_dataset/NASA_MOSFET_TEMP",MOSFET_num_ll=["dr36","dr8","dr11","dr14"])
            plt.show()
    if Temp_dR:
        # 觀測出 單顆 mosfet 再同樣溫度下量測的數值 的確隨著老化次數加多 數值有所浮動
        #36 這顆 mosfet 僅有一次run  所以 不可加入評比 只能評斷說 隨著溫度上升量到的Rth 值 會隨之上升
        # 200 度以上的溫度去烤mosfet 老化非常明顯
        df_rdson = pd.read_csv("./paper_dataset/dr_aging_800min.csv")
        # print(df_rdson.columns[0::2])#X   time
        # print(df_rdson.columns[1::2])#Y   rdson
        dict={"rdson":[],"temp":[],"run":[],"MOSFET":[],"AVG":[]}
        for  time_label,rdson_label,MOSFET_rdson,MOSFET_TEMP in zip(df_rdson.columns[0::2],df_rdson.columns[1::2],MOSFET_rdson_ll,MOSFET_TEMP_ll):
            df2 = temp_dr_analyze(read_file="./paper_dataset/NASA_MOSFET_TEMP/"+MOSFET_TEMP+".csv",temp_ll=temp_ll)
            df2=df2.set_index("run")
            arr0_last =copy.deepcopy( df2.values)#last
            arr0_duration= copy.deepcopy(df2.values)#duration
            arr_sample_x=[]
            arr_sample_y=[]
            time_tick_table = []
            time_duration=np.sum(arr0_last,axis=1)


            for i in range(arr0_last.shape[0]):
                time_tick_table = [0 if temp_data == 0 else np.sum(arr0_last[i,0:idx_col+1]) for idx_col, temp_data in enumerate(arr0_last[i,:])]
                arr0_last[i,:]=time_tick_table if i==0 else [time_duration[i-1]+j if j!=0 else 0 for j in time_tick_table]

            arr0_last=arr0_last.tolist()#last

        ######################
            df_rdson[rdson_label][0] = 0#每顆MOSFET Variation 從第0 %開始計算
            df_rdson[time_label][0] = 0 #time 從第0時刻開始計算
            df_rdson[rdson_label] = (df_rdson[rdson_label] / Nominal_resistanc) * 100
        #####################
            for idx,arr0_last_s in enumerate(arr0_last):
                # one_run=[   [l for l in range (int(last-dur),int(last))] if dur!=0 else 0 for  dur,last in zip(arr0_duration[idx,:],arr0_last_s) ]
                one_run_x = [[l for l in range(int(last - dur), int(last))] for dur, last in zip(arr0_duration[idx, :], arr0_last_s)]
                one_run_y = [np.interp([l for l in range(int(last - dur), int(last))], df_rdson[time_label], df_rdson[rdson_label])for dur, last in zip(arr0_duration[idx, :], arr0_last_s)]
                arr_sample_x.append(one_run_x)
                arr_sample_y.append(one_run_y)


            temp_ll = [40.0, 100.0, 150.0, 200.0, 210.0, 220.0, 230.0, 240.0, 250.0]

            for idx,arr_sample_y_s in enumerate(arr_sample_y):
                for rdson_s, temp_s in zip(arr_sample_y_s, temp_ll):
                    if rdson_s.size==0:
                        continue
                    else:
                        dict["rdson"].extend(rdson_s)
                        dict["rdson"].extend([mean(rdson_s).tolist()])
                        dict["AVG"].extend("N" * rdson_s.shape[0])
                        dict["AVG"].extend("Y")
                        dict["temp"].extend([temp_s] * (rdson_s.shape[0]+1))# 因為AVG 向次 所以需要+1
                        dict["run"].extend(["run%d"%idx] * (rdson_s.shape[0]+1))
                        dict["MOSFET"].extend([MOSFET_rdson] * (rdson_s.shape[0]+1))


        df3=pd.DataFrame.from_dict(dict)
        data_avg = df3[(df3.MOSFET == MOSFET_SEL) & (df3.AVG == "Y")]
        # palette_ll=['green','orange','brown','dodgerblue','red','fuchsia','purple']
        palette_ll=sns.color_palette('viridis', n_colors=len(data_avg["run"].unique()))
        palette_ll.reverse()
        color_dict = {run_name: color_s for run_name, color_s in zip(data_avg["run"].unique(), palette_ll)}

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        sns.set()
        sns.lineplot(x='temp', y='rdson',hue="run",data=data_avg,palette=color_dict,marker="o", linewidth=2, markersize=7)
        sns.stripplot(data=df3[(df3.MOSFET == MOSFET_SEL) & (df3.AVG == "N")], x="temp", y="rdson", hue="run", ax=axes[0],palette=color_dict)
        axes[0].set_title(MOSFET_SEL+" Rdson variation V.S Temperature ", fontsize=20)
        axes[0].tick_params(axis='x', labelsize=15,rotation=30)
        axes[0].tick_params(axis='y', labelsize=15,rotation=30)
        axes[0].set_xlabel("Temperature($^\circ$C)",fontsize=15)
        axes[0].set_ylabel("Rdson(%)", fontsize=15)
        axes[0].legend(fontsize=10, title="Test run", title_fontsize=15)
        axes[0].yaxis.grid()

        axes[1].set_title(MOSFET_SEL+" Average Rdson variation V.S Temperature ", fontsize=20)
        axes[1].tick_params(axis='x', labelsize=15,rotation=30)
        axes[1].tick_params(axis='y', labelsize=15,rotation=30)
        axes[1].set_xlabel("Temperature($^\circ$C)",fontsize=15)
        axes[1].set_ylabel("Rdson(%)", fontsize=15)
        axes[1].legend(data_avg["run"].unique(),fontsize=10, title="Test run", title_fontsize=15)
        axes[1].yaxis.grid()
        plt.show()
        # plt.close(fig)
        return df3
if __name__ == '__main__':
    pass
    MOSFET_SEL = 'MOSFET 36   '
    df3=execution(dR_AGING=False,Temp_dR=True,MOSFET_SEL=MOSFET_SEL,COX_GATE=False,Temp_time=False)
    # data=df3[(df3.MOSFET == MOSFET_SEL) & (df3.AVG == "N") & (df3.temp >= 200)]
    # data.reset_index(inplace=True, drop=True)
    # data.index=data.index+1
    # print(data.index)