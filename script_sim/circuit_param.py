import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
Passive_Component_Size=True
def boost_converter_param(Vo,Vi,Vd,Io,freq,delta_Vo,delta_Vi,L_coff,power_efficenicy):
    L=Vi/(L_coff*freq*Io)*(1-Vi/(Vo+Vd))*(Vi/(Vo+Vd))*power_efficenicy
    if delta_Vi !=0:
        C_in=Vi/(8*(freq**2)*L*delta_Vi)*(1-Vi/(Vo+Vd))
    else:
        C_in=0
    C_out=Io/(freq*delta_Vo)*(1-Vi/(Vo+Vd))
    return L,C_in,C_out

if Passive_Component_Size:
    Vo=42
    Vi=18
    Vd=0.875 # #查詢diode 的 datasheet
    R_load=28
    Io=Vo/R_load
    freq=200E3
    delta_Vi=100E-3
    delta_Vo=100E-3
    L_coff=0.2#delata I(L) 佔 I(L)的比例0.2~0.4
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

    print("L : {:e}".format(L))
    print("Cin : {:e}".format(C_in))
    print("Cout : {:e}".format(C_out))
    #########繪製出 freq 與其他 被動元件 電容 電桿大小之間的關係
    # 說明了 利用高頻元件的好處有助於scale down size
    # freq_ll=np.array([60E3,80E3,100E3,150E3,200E3])
    freq_ll=np.sort(np.append(np.linspace(60E3,1000E3,100),200E3))
    L_Ci_Co=list(map(lambda x: boost_converter_param(Vo=Vo,
                                           Vi=Vi,
                                           Vd=Vd,
                                           Io=Io,
                                           freq=x,
                                           delta_Vo=delta_Vo,
                                           delta_Vi=delta_Vi,
                                           L_coff=L_coff,
                                           power_efficenicy=power_efficenicy),freq_ll))
    A=np.array(L_Ci_Co)
    A=(np.transpose(A/A[0,:])-1)*100
    fig,axes=plt.subplots(1,2,figsize=(15, 8))
    color = sns.color_palette("Set1")
    meas_datas=[A[0],A[2]]
    title_ll=["Inductance Size","Output Capacitor Size"]
    y_titles=["(%)","(%)"]
    freq_ll_k=np.round(freq_ll/1E3,decimals=1)
    for idx, (meas_data, title,y_title) in enumerate(zip(meas_datas, title_ll,y_titles)):
        axes[idx].plot(freq_ll_k, meas_data, color=color[idx])
        axes[idx].plot(freq_ll_k[-1],meas_data[-1], marker="*",color="black", markersize=10)
        axes[idx].text(freq_ll_k[-1],meas_data[-1],"\n\n%.2f" %meas_data[-1] + "%\n\n", fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        axes[idx].axvline(x=200,color='k',linestyle='--')
        axes[idx].plot(200,meas_data[ np.where(freq_ll_k == 200)] ,marker=".",color="black", markersize=10)
        axes[idx].text(200,meas_data[ np.where(freq_ll_k == 200)] ,"\n\n%.2f" %meas_data[ np.where(freq_ll_k == 200)]+ "%\n\n", fontsize=12, color="k",style="italic", weight="bold", verticalalignment='center', horizontalalignment='right', rotation=0)
        axes[idx].set_xlabel( "Frequency(kHz)", fontsize=15)
        axes[idx].set_ylabel(y_title, fontsize=15)
        axes[idx].set_title(title, fontsize=20)
        axes[idx].tick_params(axis='x', labelsize=15)
        axes[idx].tick_params(axis='y', labelsize=15)
        axes[idx].set_xticks(list(axes[idx].get_xticks()) + [60])
        # axes[idx].set_xticks([60,200,400,600,800,1000])
        axes[idx].yaxis.grid()
    plt.suptitle('Frequency V.S. Passive Components Size',fontsize=25)
    plt.show()
Ciss=1940E-12
freq=200E3
Q=0.5#(slow)0.5~1(fast)
Ls=1/(Ciss*((2*math.pi*freq)**2))
Rgate=Ls/Q
print(Ls)
print(Rgate)