import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import multiprocessing
cpus = multiprocessing.cpu_count()
from multiprocessing import Process, Pool


read_file=pd.read_csv("../data/Draft3.txt",sep="\t",skiprows=2)
read_file.to_csv("../data/Draft.csv",index=None)
data=pd.read_csv("../data/Draft.csv",names=['time','Vds_sec_fet','VT_pri','VT_sec','fet_current'])



time=data.time# length 326826 for 10ms 32682.6 for 1ms
#for 5~6s scope
#1263328.4 ~ 1579160.5 => 1263328 ~ 1579161
#length=315833+1(1579161-1263328)
start_probe_time=32683*4
end_probe_time=32683*5

probe_length=end_probe_time-start_probe_time+1

core_num=6
voltage_primary=data.VT_pri
voltage_seconedary=data.VT_sec
start_time=start_probe_time-1
# start_time=0


def process_map(length):
    # length=(initial,last)
    status_process = 0
    for time_step in range(length[0],length[1]):
        status_process=status_process+1
        # try:
        k = np.format_float_scientific(float(voltage_seconedary[start_time+time_step]), unique=False, precision=2)
        voltage_seconedary[start_time+time_step] = k
        print("processing..........{} %".format(status_process*100/(length[1]-length[0])))
        # except Exception as e:
        #     print(e)
        #     print("{} timestep got error".format(time_step))
        #     os.system('pause')
    pass

def process_cut(core_num,length_t):
    process_length=length_t//core_num
    length_split_list=[]
    for i in range (core_num):
        single_list=[i*process_length,(i+1)*process_length]
        length_split_list.append(single_list)
        pass
    return length_split_list

if __name__ == '__main__' :
    pass

    split_length_list=process_cut(core_num=core_num,length_t=315834)
    print(split_length_list)

    pool = Pool(core_num)
    pool.map(process_map,split_length_list)
    print("plotting......")
    plt.plot(time[start_probe_time:end_probe_time], voltage_seconedary[start_probe_time:end_probe_time])
    print("plotting...... done")
    plt.show()