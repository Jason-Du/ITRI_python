from PyLTSpice.LTSteps import LTSpiceLogReader
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
import matplotlib.ticker as tick
from matplotlib.ticker import FormatStrFormatter
DRAW=False
data = LTSpiceLogReader("./result/r8002cnd3_MOS_N_L.log")
wanted_label=["tr1","tr2","imax"]
# "./result/r8002cnd3_MOS_N_L.log"
# "monte_SiC.log"
# print("Number of steps  :", data.step_count)
step_names = data.get_step_vars()
meas_names = data.get_measure_names()
print(step_names)
print(meas_names)
# print(step_names)
# for step in step_names:
#     print(step)# tol
# for name in meas_names:
#     print(name)# imax_FROM# imax_TO# tr1# tr1_FROM# tr1_TO# tr2# tr2_FROM# tr2_TO
# for i in range(data.step_count):
#     print([float(data[step][i])for step in step_names])
#     print([float(data[name][i])for name in meas_names])

#定義要觀察的量測指標
data_dict={}
for data_label in wanted_label:
    data_dict[data_label]=data[data_label]
step_list=[x+1 for x in data[list(step_names)[0]]] # """幅度 <0 為減少 >0 為增加"""  調整為  """變化幅度 <1 為減少 >1 為增加"""


#以第一筆資料當作比較依據 後需資料除以 第一筆資料 查看其變化幅度 <1 為減少 >1 為增加
for data_label in wanted_label:
    data_dict[data_label] = [float(x/data_dict[data_label][0]) for x in data_dict[data_label]]

# 把變化最大的資料 print 出來
#資料全部減1 獲取變化的"幅度" <0 為減少 >0 為增加
var_dict={}
var_dict["tr1"]=[ x-1 for x in data_dict["tr1"]]
var_dict["tr2"]=[ x-1 for x in data_dict["tr2"]]
var_dict["imax"]=[ x-1 for x in data_dict["imax"]]

var_max_dict={}
#找出變化最大變化的"幅度"
var_max_dict["tr1"] = max(var_dict["tr1"],key=abs)
var_max_dict["tr2"] = max(var_dict["tr2"],key=abs)
var_max_dict["imax"] = max(var_dict["imax"],key=abs)

print(" rise time increase rate  : {} %".format( (  var_max_dict["tr1"]*100) )  )
print(" fall time increase rate  : {} %".format( ( var_max_dict["tr2"]*100) )  )
print(" maximum id current  increase rate : {} % ".format( ( var_max_dict["imax"]*100) )  )

# 把變化最大的資料 的index 儲存起來
rise_time_idx=var_dict["tr1"].index(var_max_dict["tr1"])
fall_time_idx=var_dict["tr2"].index(var_max_dict["tr2"])
max_id_idx=var_dict["imax"].index(var_max_dict["imax"])

# 繪製曲線圖 橫縱軸都是以增加幅度繪製




if DRAW:
    y_low=0.9
    y_high=1.3
    fig,axes=plt.subplots(1,3,figsize=(20, 10))
    #繪製模擬時間內的最大電流
    axes[0].plot(step_list,data_dict["imax"],color='red' )
    axes[0].plot(step_list[max_id_idx],data_dict["imax"][max_id_idx],marker="*",color="black",markersize=10)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_xlabel("Cap variation(rate)",fontsize=15)
    axes[0].set_xlabel("Junction potential variation(rate)",fontsize=15)
    axes[0].set_title("ID current maximum",fontsize=15)
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    # axes[0].set_ylim(min(data_dict["imax"]),max(data_dict["imax"]))
    axes[0].set_xlim(1.00,2.00)
    axes[0].set_ylim(y_low,y_high)
    #繪製模擬時間內的rise time
    axes[1].plot(step_list,data_dict["tr1"],color='blue')
    axes[1].plot(step_list[rise_time_idx],data_dict["tr1"][rise_time_idx],marker="*",color="black",markersize=10)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].xaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_xlabel("Cap variation(rate)",fontsize=15)
    axes[1].set_xlabel("Junction potential variation(rate)",fontsize=15)
    axes[1].set_title("rise time ",fontsize=20)
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    # axes[1].set_ylim(min(data_dict["tr1"]),max(data_dict["tr1"]))
    axes[1].set_ylim(y_low,y_high)
    #繪製模擬時間內的fall time
    axes[2].plot(step_list,data_dict["tr2"],color='blue')
    axes[2].plot(step_list[fall_time_idx],data_dict["tr2"][fall_time_idx],marker="*",color="black",markersize=10)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].xaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_xlabel("Cap variation(rate)",fontsize=15)
    axes[2].set_xlabel("Junction potential variation(rate)",fontsize=15)
    axes[2].set_title("fall time ",fontsize=20)
    axes[2].tick_params(axis='x', labelsize=15)
    axes[2].tick_params(axis='y', labelsize=15)
    # axes[2].set_ylim(min(data_dict["tr2"]),max(data_dict["tr2"]))
    axes[2].set_ylim(y_low,y_high)
    
    plt.show()