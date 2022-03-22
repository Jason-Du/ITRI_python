#若是想要使用 Takagi PAPER https://www.mdpi.com/1996-1073/14/8/2135/htm 的方法 就必須使用多筆的歷史資料
#將歷史資料先進行分類建模 然後再進行模型選擇預測
from filterpy.kalman import KalmanFilter
import numpy as np
from filterpy.common import Q_discrete_white_noise
from aging_rate import execution
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import math
temp_ll=[40.0,100.0,150.0,200.0,210.0,220.0,230.0,240.0,250.0]
MOSFET_SEL = 'MOSFET 8'
failure_value=80
Predict_curve=False
TTF=True
def curve_func(x,a,b):
    pass
    return a*x*np.log(x*b)
if Predict_curve:
    df3 = execution(
        Temp_dR=True,
        AF_COFF=False,
        COX_GATE=False,
        Vth_AGING=False,
        dR_AGING=False,
        Temp_time=False,
        Nominal_resistanc=0.25,
        MOSFET_rdson_ll=['MOSFET 14', 'MOSFET 8', 'MOSFET 9', 'MOSFET 36', 'MOSFET 11', 'MOSFET 12'],
        MOSFET_TEMP_ll=["dr14", "dr8", "dr9", "dr36", "dr11", "dr12"],
        MOSFET_SEL=MOSFET_SEL,
        temp_ll=temp_ll
    )
    data = df3[(df3.MOSFET == MOSFET_SEL) & (df3.AVG == "N") & (df3.temp >= 200)]
    data.reset_index(inplace=True, drop=True)
    data.index = data.index + 1
    print(data)
    input_step = data.index[-1] - 60
    # -1為不進行預測
    # data.index[-1]-60 謥最後一筆到算回來 60為要預測的步數
    sns.set(context='notebook', style='whitegrid')
    color=sns.color_palette("tab10")
    
    fig2, axes2 = plt.subplots(1, 1, figsize=(20, 10))
    x1=np.array(data.index)
    y1=data["rdson"]

    # result=curve_fit(f=curve_func,xdata=list(data.index)[0:input_step],  ydata=data["rdson"][0:input_step],  p0=(4, 0.1))
    # y2=curve_func(x=x1,a=result[0][0],b=result[0][1])

    # polynomial 方程式 全部data 最為 input 五次式 效果最好
    parameter=np.polyfit(x1[0:input_step],y1[0:input_step],5)
    print(parameter)
    y2=np.array([parameter_s*(x1**idx) for idx,parameter_s in enumerate(parameter[::-1])])
    y2=np.sum(y2,axis=0)



    sns.scatterplot(x=x1,y=y1,color=color[0],label=MOSFET_SEL,ax=axes2)  # 繪製所有曲線
    axes2.plot(x1[0:input_step],y2[0:input_step],color=color[1],label=MOSFET_SEL+" Trend Line")
    if input_step!=-1:
        axes2.plot(x1[input_step:],y2[input_step:],color=color[2],label=MOSFET_SEL+" Prediction Curve",linestyle='--')
    axes2.set_title("Power MOSFET \u0394R Variation Prediction T=210~250$^\circ$C Vgs=10V Freq=1kHz VDS=4V ", fontsize=20)
    axes2.set_xlabel("Aging Time (min)", fontsize=15)
    axes2.set_ylabel("\u0394R variation(%)", fontsize=15)
    axes2.tick_params(axis='x', labelsize=15)
    axes2.tick_params(axis='y', labelsize=15)
    axes2.legend()
    axes2.yaxis.grid()
    plt.show()
if TTF:
    # Time to Failure plot base on the prediction verses the real condition
    pass
    df3 = execution(
            Temp_dR=True,
            AF_COFF=False,
            COX_GATE=False,
            Vth_AGING=False,
            dR_AGING=False,
            Temp_time=False,
            Nominal_resistanc=0.25,
            MOSFET_rdson_ll=['MOSFET 14', 'MOSFET 8', 'MOSFET 9', 'MOSFET 36', 'MOSFET 11', 'MOSFET 12'],
            MOSFET_TEMP_ll=["dr14", "dr8", "dr9", "dr36", "dr11", "dr12"],
            MOSFET_SEL=MOSFET_SEL,
            temp_ll=temp_ll
        )
    data = df3[(df3.MOSFET == MOSFET_SEL) & (df3.AVG == "N") & (df3.temp >= 200)]
    data.reset_index(inplace=True, drop=True)
    data.index = data.index + 1

    input_step = 250 #開始預測的時間點
    # -1為不進行預測
    # data.index[-1]-60 謥最後一筆到算回來 60為要預測的步數
    sns.set(style='whitegrid')
    color=sns.color_palette("tab10")
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 10))
    x1=np.array(data.index)
    y1=data["rdson"]


    #套用各種 model predict 的地方
    y_ll = []
    # 可使用exp scale 的方式去 且 用 log 當作error 評估方式 去得到 trend line
    # result=curve_fit(f=curve_func,xdata=list(data.index)[0:input_step],  ydata=data["rdson"][0:input_step],  p0=(4, 0.1))
    # y2=curve_func(x=x1,a=result[0][0],b=result[0][1])
    # polynomial 方程式 全部data 最為 input 五次式 效果最好
    parameter=np.polyfit(x1[0:input_step],y1[0:input_step],5)
    y2=np.array([parameter_s*(x1**idx) for idx,parameter_s in enumerate(parameter[::-1])])
    y2=np.sum(y2,axis=0)
    y_ll.append(y2)

    predict_model_ll = ["Polynomial of degree 5"]
    thres_index_real    = np.where(y1>=80)[0][0]
    thres_index_predict_ll = [np.where(y_s>=80)[0][0] for y_s in y_ll ]


    sns.scatterplot(x=x1,y=y1,color=color[0],label=MOSFET_SEL,ax=axes2[0])  # 繪製所有曲線
    # axes2[0].plot(x1[0:input_step],y2[0:input_step],color=color[1],label=MOSFET_SEL+" Trend Line")
    for idx,(predict_model,thres_index_predict) in enumerate(zip(predict_model_ll,thres_index_predict_ll)):
        axes2[0].plot(x1[input_step:thres_index_predict],y2[input_step:thres_index_predict],color=color[idx+1],label=predict_model,linestyle='-')
    axes2[0].axvline(x=input_step, color='r', linestyle='--',label="Prediction Start")
    axes2[0].set_title("Power MOSFET  \u0394R Variation Prediction ", fontsize=20)
    axes2[0].set_xlabel("Aging Time (min)", fontsize=15)
    axes2[0].set_ylabel("\u0394R variation(%)", fontsize=15)
    axes2[0].tick_params(axis='x', labelsize=15)
    axes2[0].tick_params(axis='y', labelsize=15)
    axes2[0].legend()
    axes2[0].grid(axis='x')
    #
    axes2[1].plot([x1[input_step],x1[thres_index_real]],[y1[thres_index_real],0], color=color[0],label="Real Curve")
    axes2[1].axvline(x=input_step, color='r', linestyle='--', label="Prediction Start")
    for idx, (predict_model, thres_index_predict) in enumerate(zip(predict_model_ll, thres_index_predict_ll)):
        axes2[1].plot([x1[input_step],x1[thres_index_predict]], [y2[thres_index_predict],0],color=color[idx+1], label=predict_model)
    axes2[1].set_title("Power MOSFET  Time To Failure (\u0394R>%d%%)"%failure_value, fontsize=20)
    axes2[1].set_xlabel("Time (min)", fontsize=15)
    axes2[1].set_ylabel("Time To Failure(min)", fontsize=15)
    axes2[1].tick_params(axis='x', labelsize=15)
    axes2[1].tick_params(axis='y', labelsize=15)
    axes2[1].legend()
    axes2[1].grid(axis='x')

    plt.show()


