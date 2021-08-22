import ltspice
import matplotlib.pyplot as plt
import numpy as np
import os
# 315712 point in 10 ms
# 31571.2
# 取 5-6ms
# 126284.8~~~157856
# 126285-157856

l = ltspice.Ltspice(os.path.dirname(__file__)+'\\Draft3.raw')
# Make sure that the .raw file is located in the correct path
l.parse() 

time = l.get_time()
print(len(time))
print(time[126285:157856])
# print(np.max(time))
# print(type(time))
# V_source = l.get_data('V(n007)')
V_cap = l.get_data('V(N009,N010)')

# print(type(V_source))
# print(len(V_source))
# print(np.min(V_source))


#plt.plot(time[126285:157856], V_source[126285:157856])
plt.scatter(time[126285:157856],V_cap[126285:157856],marker='+')
# plt.plot(time, V_cap)
plt.show()
