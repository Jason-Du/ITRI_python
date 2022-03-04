


def boost_converter_param(Vo,Vi,Vd,Io,freq,delta_Vo,delta_Vi,L_coff,power_efficenicy):
    L=Vi/(L_coff*freq*Io)*(1-Vi/(Vo+Vd))*(Vi/(Vo+Vd))*power_efficenicy
    C_in=Vi/(8*(freq**2)*L*delta_Vi)*(1-Vi/(Vo+Vd))
    C_out=Io/(freq*delta_Vo)*(1-Vi/(Vo+Vd))
    return L,C_in,C_out
Vo=5
Vi=3.3
Vd=0.5
Io=1.667
freq=300E3
delta_Vi=30E-3
delta_Vo=50E-3
L_coff=0.4#delata I(L) 佔 I(L)的比例
power_efficenicy=1.0

L, C_in, C_out=boost_converter_param(Vo=Vo,
                                     Vi=Vi,
                                     Vd=Vd,
                                     Io=Io,
                                     freq=freq,
                                     delta_Vo=delta_Vo,
                                     delta_Vi=delta_Vi,
                                     L_coff=L_coff,
                                     power_efficenicy=power_efficenicy)
print(L)
print(C_in)
print(C_out)
