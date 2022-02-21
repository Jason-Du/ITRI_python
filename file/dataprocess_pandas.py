import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import AutoLocator
import matplotlib.ticker as tick
from matplotlib.ticker import FormatStrFormatter
import re
log_data  = open("../script_sim/result/r8002cnd3_MOS_N_L.log", 'r')
step_num=100 #step=n 實際模擬次數為n+1 注意!!!!!!!!!!!!!!!!!!!!!
step_list = []
measure_name_list=[]
measure_single_list=[]
measure_data_list=[]
test_list=[]
measure_count=0
data_dict={}

for line in log_data:
    # print(line)
#   spice 裡 的變化幅度
    pattern=re.compile(r".step.*=(.+)")
    step_str=re.search(pattern,line)
    if step_str is not None:
        if step_str.group(1)==123:
            pass
            step_list.append(0)
        else:
            step_list.append(float(step_str.group(1))+1)

#   Measurement: imax 偵測到開始有資料要收集
    pattern2 = re.compile(r"Measurement: (.+)")
    measure_dect = re.search(pattern2, line)

    if measure_dect is not None:
        measure_name_list.append(measure_dect.group(1))
#   step	你要的資料	FROM	TO 第二column資料萃取
    pattern3 = re.compile(r"\s+(\d+)\s+(.+)\s+(.+)\s+(.+)")
    measure_value = re.search(pattern3, line)
    if measure_value is not None:
        measure_count=measure_count+1
        tmp=measure_value.group(2)
        measure_single_list.append(tmp)

# 每完成全部 step 資料的收集就將該 list 存起來到 measure_data_list
    if (measure_count==step_num+1):
        measure_data_list.append(measure_single_list)
        measure_single_list = []
        measure_count = 0
# 建立 dataset dictionary

for index,measure_name in enumerate(measure_name_list):
    pass
    data_dict[measure_name]=measure_data_list[index]

# floating 化 所有數據資料
for data_label in sorted(data_dict):
    pass
    data_dict[data_label] = [float(x) for x in data_dict[data_label]]
#以第一筆資料當作比較依據 後需資料除以 第一筆資料 查看其變化幅度 <1 為減少 >1 為增加
for data_label in sorted(data_dict):
    data_dict[data_label] = [float(x/data_dict[data_label][0]) for x in data_dict[data_label]]
print(step_list)
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


