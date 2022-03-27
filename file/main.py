import ltspice
import matplotlib.pyplot as plt
import numpy as np
import os
# 315712 point in 10 ms
# 31571.2
# 取 5-6ms
# 126284.8~~~157856
# 126285-157856

# l = ltspice.Ltspice(os.path.dirname(__file__)+'\\file/monte_SiC.raw')
# Make sure that the .raw file is located in the correct path
# l.parse()

# time = l.get_time()
# print(len(time))
# print(time[126285:157856])
# print(np.max(time))
# print(type(time))
# V_source = l.get_data('V(n007)')
# V_cap = l.get_data('V(N009,N010)')

# print(type(V_source))
# print(len(V_source))
# print(np.min(V_source))


# #plt.plot(time[126285:157856], V_source[126285:157856])
# plt.scatter(time[126285:157856],V_cap[126285:157856],marker='+')
# # plt.plot(time, V_cap)
# plt.show()
# lst = [{'id':'1234','name':'Jason'}, {'id':'2345','name':'Tom'}, {'id':'3456','name':'Tom'}]
# for (index, d) in enumerate(lst):
#     if d["name"] == "Tom":
#         print(index)
# tom_index = next((index for (index, d) in enumerate(lst) if d["name"] == "Tom"),None)
# # tom_index = next((index for (index, d) in enumerate(lst) if d["name"] == "Tom"), None)
# print(tom_index)
# 1
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# titanic_dataset = sns.load_dataset("titanic")
# print(type(titanic_dataset))
# print(titanic_dataset)
# print(titanic_dataset.keys())
# print(type(titanic_dataset["survived"]))
# sns.barplot(x = "class", y = "survived", hue = "embark_town", data = titanic_dataset,ci=None)
# plt.show()
# # hue= measure_name
# x=   param name
#y= measure value
# def aging_AF_coff(A,B,C,Temp,Vgate,a):
#     pass
#     AF_coff=A*(10**(C*Temp))*(10**(B*Vgate+a))
#     return AF_coff
# import numpy as np
# A=np.linspace(20,40,5)
# B=np.linspace(50,100,10)
# C=np.tile(A,(10,1))
# D=np.transpose(np.tile(B,(5,1)))
# E=aging_AF_coff(A=-17.2,B=0.4,C=-0.15,Temp=D,Vgate=C,a=0)
# print(E.shape)
# fig,axes=plt.subplots(1,1,figsize=(20, 10))
# # load dataset
# print(1/(260E-9))
# print(1/(31E-9))
# print(np.exp(6))
# shape=np.array([1.2567,1.0727,1.45316]).reshape(1,-1)
# scale=np.array([82.1,148,89.2]).reshape(1,-1)
# x = np.linspace(0,3E2,1000).reshape(1,-1)
# print(shape.shape)
# print(x.shape)
# print((scale.T*(x*shape.T)).shape)
# print(np.log(10))
print(6.13E-6/1.6061E-5)