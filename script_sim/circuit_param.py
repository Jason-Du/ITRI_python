import numpy as np
import matplotlib.pyplot as plt


def boost_converter_param(Vo,Vi,Vd,Io,freq,delta_Vo,delta_Vi,L_coff,power_efficenicy):
    L=Vi/(L_coff*freq*Io)*(1-Vi/(Vo+Vd))*(Vi/(Vo+Vd))*power_efficenicy
    if delta_Vi !=0:
        C_in=Vi/(8*(freq**2)*L*delta_Vi)*(1-Vi/(Vo+Vd))
    else:
        C_in=0
    C_out=Io/(freq*delta_Vo)*(1-Vi/(Vo+Vd))
    return L,C_in,C_out
def LT1619(Vo,R2):
    R1=R2*((Vo/1.24)-1)
    return R1,R2


Vo=42
Vi=18
Vd=0.875 # #查詢diode 的 datasheet
Io=42/28
freq=260E3
delta_Vi=50E-3
delta_Vo=50E-3
L_coff=0.4#delata I(L) 佔 I(L)的比例0.2~0.4
power_efficenicy=0.95

L, C_in, C_out=boost_converter_param(Vo=Vo,
                                     Vi=Vi,
                                     Vd=Vd,
                                     Io=Io,
                                     freq=freq,
                                     delta_Vo=delta_Vo,
                                     delta_Vi=delta_Vi,
                                     L_coff=L_coff,
                                     power_efficenicy=power_efficenicy)
# print("{:e}".format(L))
# print("{:e}".format(C_in))
# print("{:e}".format(C_out))


