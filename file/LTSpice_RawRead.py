from PyLTSpice.LTSpice_RawRead import LTSpiceRawRead
import numpy as np
from matplotlib import pyplot as plt

LTR = LTSpiceRawRead(r"monte_SiC.raw")
#monte_SiC.raw
print(LTR.get_trace_names())
print(LTR.get_raw_property())

# IR1 = LTR.get_trace("I(V1)")
# x = LTR.get_trace('time')  # Gets the time axis
# steps = LTR.get_steps()
# ID_Cur_arr=IR1.get_wave(0)
# ID_Cur_arr_len=len(ID_Cur_arr)

# print(np.amin(ID_Cur_arr[0:int(ID_Cur_arr_len/2)]))
# print(np.amin(ID_Cur_arr[int(ID_Cur_arr_len/2):]))
# 取線段中的一點對應的Y值 但維杜太小 python 抓取不到 改用spice 內部語法抓取##############
# start_time =np.interp(-6.231389999389648*0.9,ID_Cur_arr[0:int(ID_Cur_arr_len/2)],x.get_time_axis(0)[0:int(ID_Cur_arr_len/2)])
# finish_time=np.interp(-6.231389999389648*0.1,ID_Cur_arr[0:int(ID_Cur_arr_len/2)],x.get_time_axis(0)[0:int(ID_Cur_arr_len/2)])
# ####################################################################
# for step in range(len(steps)):
#
#
#    plt.plot(x.get_time_axis(step), -1*IR1.get_wave(step))
# # plt.legend()  # order a legend
# plt.show()