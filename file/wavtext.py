import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
read_file=pd.read_csv("Draft3.txt",sep="\t",skiprows=2)
read_file.to_csv("Draft.csv",index=None)
data=pd.read_csv("Draft.csv",names=['time','V2xIR1','V2','V3','V4','V5','V6','V7','V9','IL1','IR1'])

time=data.time# length 3185321 for 10ms 315832.1 for 1ms
#for 5~6s scope
#1263328.4 ~ 1579160.5 => 1263328 ~ 1579161
voltage_primary=data.V7
voltage_seconedary=data.V9
for time_step in range(0,len(time)):
    try:
        k=np.format_float_scientific(float(voltage_seconedary[time_step]),unique=False,precision=2)
        voltage_seconedary[time_step]=k
    except:
        print("{} timestep got error".format(time_step))
plt.scatter(time[1263328:1579160],voltage_seconedary[1263328:1579160],marker='+')
plt.show()




