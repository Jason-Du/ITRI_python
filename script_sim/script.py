"""
功能 : 反覆修改 spice lib  檔 並使用 ltspice 模擬
parsing 能做到的事 事修改特定一行資料 進行 rewrite
for 單個 parameter 的模擬
每次修改 必須修改兩處 一是要進行 step 模擬的 param 二是上次修改的param 須改回來
{
    modelname :
    [
        {
            param_name:
            param_idx:
            param_value:
        }
    ]
}
"""
from PyLTSpice.LTSpiceBatch import SimCommander
import re
import os
import shutil
from itertools import permutations
from os import listdir
from os.path import isfile, isdir, join
CLEAR_LIB_FILE=True
CLEAR_RESULT_FILE=False
param_num=2# LTSPICE 電路檔記得要調整

def spice_data_anlyz(input_lib=""):
    spice_txt = open(input_lib,'r') #放置lib 檔的地方
    pars=[] # 參數集合 小單元為 par_dict[{'par_name':///, 'par_val':///, 'idx':///}] #於for 迴圈中使用
    par_dict={} # 包含 param 的名子 值 以及 位在哪一行 供之後可以直接替換掉
    model_names=[] # model 的名稱集 亦是 mod_par_dict 的 keys
    last_pars=[] # 上次修改的參數名稱 [{'par_name': 'VJ', 'par_val': '1.94384', 'idx': 46}] 小單位為 last_par_dict
    mod_par_dict={} # key是 model的名稱 value 是 param 的名稱 小單元為 pars "model_name":[(pars)],"":[]
    flag=0
    for line_idx,line in enumerate(spice_txt):
        pass
        # pattern=re.compile(r"r"+"\s*(.+)*=(.+)")
        pattern1 = re.compile(r".MODEL\s(\w+)\s")
        model_name=re.search(pattern1,line)
        pattern2=re.compile(r"\s(\w+)*=(.+)")
        par=re.search(pattern2,line)
        pattern3=re.compile(r"\s(\w+)*={(.+)\*")#正在inject aging 的param
        last_par=re.search(pattern3,line)
        pattern4=re.compile(r".ENDS")
        end_bool=re.search(pattern4,line)


        if model_name is not None :
            model_names.append(model_name.group(1))


        if (model_name is not None or end_bool is not None):
            if(flag > 0):
                mod_par_dict[model_names[flag-1]]=pars
            flag=flag+1
            pars = []

        if par is not None and last_par is None and par.group(1)!="LEVEL": # last_par is None 條件是將 {//////*(1+tol)} 給予 if last_par is not None: # 程式碼去處理 應蓋也要納入 model 作為條件
            # 有甚麼參數是不需要納入考量的也可以寫在這裡 如 LEVEL
            par_dict["par_name"]=par.group(1)
            par_dict["par_val"] = par.group(2)
            par_dict["idx"]=line_idx
            pars.append(par_dict)
            par_dict={}

        if last_par is not None: #處李{//////*(1+tol)}pattern 萃取出 //////
            par_dict["par_name"] = last_par.group(1)
            par_dict["par_val"] =last_par.group(2)
            par_dict["idx"] =line_idx
            last_pars.append(par_dict)
            pars.append(par_dict)# pattern2 無法萃取出該值 故由這裡來包辦
            par_dict={}
    spice_txt.close()

    return last_pars,mod_par_dict

def ajst_lib_revi_retu_idx(mod_par_dict={},revi_pars=[],retu_pars=[]):
    """
    # 原mosfet 參數的集合 {model name :[{par_name:\\\ , par_value\\\ , idx:\\\] }
    revi_pars是要這次改動的param list [{mod_nam : \\\ , par_name:\\\}]
    retu_pars 是要更改回來的 param list
    return 2個參數 一個為要修改的參數在 該model key 下的list 排在哪一個idx 另一個是要改回來的 小單位為[{'MOS_N': 3}, {'MOS_N': 4}]
    now_par_idxs [{'DDS': 0}]
    last_par_idxs [{'MOS_N': 3}, {'MOS_N': 4}]
    """
    # 參數需要調整回來
    now_par_idxs=[]
    last_par_idxs = []
    now_par_idx=None
    for revi_par in revi_pars:# 可能不只一個參數需要調整
        now_par_idx = next((index for (index, d) in enumerate(mod_par_dict[revi_par["mod_name"]]) if d["par_name"] == revi_par["par_name"]), None)
        now_par_idxs.append({revi_par["mod_name"]:now_par_idx})
    if retu_pars is not []:# 第一筆資料不會有修正項
        for retu_par in retu_pars:# 可能不只一個參數需要調整
            last_par_idx = next((index for (index, d) in enumerate(mod_par_dict[retu_par["mod_name"]]) if d["par_name"] == retu_par["par_name"]), None)
            last_par_idxs.append({retu_par["mod_name"]:last_par_idx})
    return now_par_idxs,last_par_idxs

# test_retu = [{"mod_name": 'MOS_N', "par_name": "RS"}, {"mod_name": 'MOS_N', "par_name": "RD"}]
# test_revi = [{"mod_name": 'DDS', "par_name": "IS"}]
# now_par_idxs, last_par_idxs = ajst_lib_revi_retu_idx(mod_par_dict=mod_par_dict,
#                                                      retu_pars=test_retu,
#                                                      revi_pars=test_revi)
# now_par_idxs=[{'DDS': 0}]
# last_par_idxs=[{'MOS_N': 3}, {'MOS_N': 4}]
#
# ajst_lib_content(mod_par_dict=mod_par_dict,
#                  now_par_idxs=now_par_idxs,
#                  last_par_idxs=last_par_idxs,
#                  output_file="test.lib",
#                  input_lib=input_lib)
def ajst_lib_content(mod_par_dict={},now_par_idxs=[],input_lib=""):
    pass

    spice_txt = open(input_lib, 'r+')
    string_list = spice_txt.readlines()
    output_file_str=""

    for idx,now_par_idx in enumerate(now_par_idxs): ##改動的內容 ={*(1+tol)}}
        output_file_str=""
        now_par_info=mod_par_dict[list(now_par_idx.keys())[0]][now_par_idx[list(now_par_idx.keys())[0]]] #前面一項是解碼出選擇哪個model 的參數 'MOS_N'後者是利用index定位出 param 的位址
        string_list[now_par_info["idx"]] = "+ {}={{{}*(1+tol{})}}\n".format(now_par_info['par_name'],now_par_info['par_val'],idx+1)
    # if last_par_idxs is not []: # 第一筆資料不會有修正項
    #     for last_par_idx in last_par_idxs:
    #         last_par_info=mod_par_dict[list(last_par_idx.keys())[0]][last_par_idx[list(last_par_idx.keys())[0]]]
    #         string_list[last_par_info["idx"]] = "+ {}={}\n".format(last_par_info['par_name'],last_par_info['par_val'])
        output_file_str = output_file_str.join(["_" + str(list(now_par_idx.keys())[0]) + "_" + str(now_par_idx["par_name"]) for now_par_idx in now_par_idxs])
    spice_txt.close()

    output_file="./lib/{}/r8002cnd3".format(param_num)+output_file_str+".lib"
    spice_txt = open(output_file, 'a+')
    spice_txt.write("".join(string_list))
    spice_txt.close()



if __name__ == '__main__':
    if CLEAR_LIB_FILE:
        path_to_dir = "./lib/{}".format(param_num)
        if (os.path.isdir(path_to_dir)):
            shutil.rmtree(path_to_dir)
        os.mkdir(path_to_dir)
    if CLEAR_RESULT_FILE:
        path_to_dir = "./result/{}".format(param_num)
        if (os.path.isdir(path_to_dir)):
            shutil.rmtree(path_to_dir)
        os.mkdir(path_to_dir)

    input_lib = r"C:\Users\user\Documents\LTspiceXVII\file\LIB\r8002cnd3_copy.lib"
    last_pars, mod_par_dict=spice_data_anlyz(input_lib=input_lib)

    now_par_idxs_ll=[]
    now_par_ll     =[]
    # 單筆參數當成parameter injection
    # for model in mod_par_dict.keys():
    #     now_par_idxs_ll=now_par_idxs_ll+[[{model:param_index,"par_name":param_info["par_name"]}]for param_index,param_info in enumerate(mod_par_dict[model])]
    for model in mod_par_dict.keys():
        now_par_idxs_ll=now_par_idxs_ll+[{model:param_index,"par_name":param_info["par_name"]}for param_index,param_info in enumerate(mod_par_dict[model])]

    now_par_idxs_ll=list(permutations(now_par_idxs_ll,2))

    for index,now_par_idxs in enumerate(now_par_idxs_ll):
        ajst_lib_content(mod_par_dict=mod_par_dict,
                         now_par_idxs=now_par_idxs,
                         input_lib=input_lib)
    # files = listdir("./lib/{}".format(param_num))
    # for f_idx,f in enumerate(files):
    #     print("{} IS UNDER SIMULATION...................{} % ".format(f,float((f_idx+1)/len(files)*100)))
    #     shutil.copyfile("./lib/"+f,r"C:\Users\user\Documents\LTspiceXVII\file\LIB\r8002cnd3.lib")
    #     pattern=re.compile(r"(.+).lib")
    #     log_name= re.search(pattern,f)
    #     LTC = SimCommander(r"C:\Users\user\Documents\LTspiceXVII\file\Montecarlo\monte_SiC.asc")
    #     LTC.run()
    #     LTC.wait_completion()
    #     shutil.copyfile(r"C:\Users\user\Documents\LTspiceXVII\file\Montecarlo\monte_SiC_1.log", "./result/{}/".format(param_num)+str(log_name.group(1))+".log")
    #     break
    ##scripting 跑 ltspice 模擬
