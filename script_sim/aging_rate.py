import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as matloader
def aging_AF_coff(A,B,C,Temp,Vgate,a):
    pass
    AF_coff=A*(10**(C*Temp))*(10**(B*Vgate+a))
    return AF_coff
# 藉由 aging 溫度 Gate 端偏壓 去計算出AF cofficient
# 範例中圍論文提到藉由 IRF4XX 系列 實驗數據結果可以得到的 A,B,C 的參數值

AF_COFF=False
COX_GATE=False
Vth_AGING=False
dR_AGING=True
def Calculate_MTTF(hours):
    return hours/(24*365)

print(Calculate_MTTF(hours=446091))

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
    X_list=['AgingMOSFET0X', 'AgingMOSFET2X','AgingMOSFET5X','AgingMOSFET6X','AgingMOSFET7X']
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
    df = pd.read_csv("./paper_dataset/dr_aging.csv")
    labels = ['AgingMOSFET 0', 'AgingMOSFET 1','AgingMOSFET 2', 'AgingMOSFET 3','AgingMOSFET 4','AgingMOSFET 5']
    X_list=['AgingMOSFET0X', 'AgingMOSFET1X','AgingMOSFET2X','AgingMOSFET3X','AgingMOSFET4X','AgingMOSFET5X']
    Y_list = ['AgingMOSFET0Y', 'AgingMOSFET1Y', 'AgingMOSFET2Y', 'AgingMOSFET3Y','AgingMOSFET4Y','AgingMOSFET5Y']




