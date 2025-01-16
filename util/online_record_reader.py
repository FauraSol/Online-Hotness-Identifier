from os import wait
import pathlib
import math
import numpy as np
from collections import OrderedDict
from typing import Optional
from sortedcontainers import SortedList
import sys

from river import datasets
from river import tree
from river import metrics

def get_page_id(address:int, size:int):
    begin_page = address // 4096
    end_page = (address + size - 1) // 4096
    return begin_page, end_page

def default_hot_gunction(item,threshold) -> bool:
    return item["access"] / (item["last_access"]-item["enqueue_instr"]) >= threshold

#可以用指数衰减热度
#可以跑一遍以8-2定律定义热页面比例

#要处理的是训练集涉及的页面的每一次访问
#出EQ的时候，给出一个判断冷/热，提供一个可以learn_one的(x,y)
#出EQ的时机：
# 1.被重复访问且被判断为热 
# 2.因容量有限被挤出去
class Evaluation_queue:
    def __init__(self, max_size,hot_thred = None,func_hot_judge = None, training:bool=False,alpha:float=-0.03,heating:int=200,waiting=10000):
        self.max_size = max_size
        self.item_dict = OrderedDict()
        self.ts = 0
        # self.func_hot_judge = func_hot_judge if func_hot_judge is not None else default_hot_gunction #热度维护函数
        self.hot_thred = hot_thred
        self.training = training
        self.alpha = alpha
        self.heating = heating
        self.waiting = waiting
        if self.training:
            self.hot_list = SortedList()
            self.backup_hot_list = SortedList()
            self.hot_list_cap = 50 * self.max_size


    def __str__(self):
        return self.item_dict.__str__()

    def heat_calculation(self, last_heat:float, access:bool, last_ts:int, cur_ts:int):
        ans = math.exp((cur_ts-last_ts)*self.alpha)*last_heat + (self.heating if access else 0)
        return ans

    @property
    def p80hot(self):
        try:
            return self.hot_list[int(0.8*len(self.hot_list))]
        except IndexError:
            print(f"hot list length: {len(self.hot_list)}, thred: {0.8*len(self.hot_list)}")
            return self.hot_thred

    @property
    def MAX_TS(self):
        return 1,000,000,000

    def enqueue(self, item) -> Optional[dict]:
        item_key = item["page_id"]
        return_val = None
        self.ts = item["n_instr"]
        if item_key in self.item_dict:
            try:
                new_heat = self.heat_calculation(self.item_dict[item_key]["heat"],True,self.item_dict[item_key]["last_access"],self.ts)
                self.item_dict[item_key]["heat"] = new_heat
                self.item_dict[item_key]["last_access"] = self.ts
            except OverflowError:
                last_heat = self.item_dict[item_key]['heat']
                last_ts = self.item_dict[item_key]['last_access']
                cur_ts = self.ts
                exp = math.exp((cur_ts-last_ts)*self.alpha)*last_heat
                print("overflow error")
                print(f"last_heat:{last_heat}")
                print(f"last_ts:{last_ts}")
                print(f"cur_ts:{self.ts}")
                print(f"exp:{exp}")

            first_key = list(self.item_dict.keys())[0]
            first_value = self.item_dict[first_key]
            if self.ts - first_value["init_access"] > self.waiting:
                return_val = self.dequeue()
            # if self.func_hot_judge(self.item_dict[item_key],self.hot_thred):
            #     return_val = {"page_id":item_key,"statistics":self.item_dict[item_key], "isHot":1}
            #     self.item_dict.pop(item_key)
            return return_val
        elif len(self.item_dict) == self.max_size:
            return_val = self.dequeue()
        self.item_dict[item_key] = {"heat":self.heating,"init_access":self.ts, "last_access":self.ts,"item":item}
        return return_val

    def dequeue(self) -> Optional[dict]:
        kv = self.item_dict.popitem(last=False)
        new_heat = self.heat_calculation(kv[1]["heat"],False,kv[1]["last_access"],self.ts)
        kv[1]["heat"] = new_heat
        kv[1]["last_access"] = self.ts
        isHot = kv[1]["heat"] > self.p80hot
        item = kv[1]
        if self.training:
            self.hot_list.add(item["heat"])
            if len(self.hot_list) > self.hot_list_cap:
                self.backup_hot_list.add(item["heat"])
                if len(self.backup_hot_list) > self.hot_list_cap:
                    self.hot_list = self.backup_hot_list
                    self.backup_hot_list = SortedList()
        # self.hot_thred = self.hot_list[int(0.8*len(self.hot_list))]
        return (kv[1], isHot)

    def get_item_by_id(self, item_id):
        return self.item_dict.get(item_id)


class Record_reader:
    def __init__(self, name:str=None, dir:str = None,eq_capacity:int = 100, hot_thred:float = 0.134,training:bool=False, outname:str = None, outdir:str = None,warmup_num: int = 1000,alpha:float=-0.0003,heating:int=200,waiting=10000) -> None:
        self.filename = name
        self.directory = dir
        self.thread_idx = -1
        self.func_idx = -1
        self.outdir = outdir
        self.outname = outname
        self.record_list = []
        self.eq = Evaluation_queue(max_size=eq_capacity,hot_thred=hot_thred,func_hot_judge=None,training=True,alpha=alpha,heating=heating,waiting=waiting)
        print(f"capacity: {eq_capacity}, thred: {hot_thred}")
        self.return_list = []
        self.warmup_num = warmup_num
        self.last_pc = 0
        self.last_address = 0
        self.ts = 0
        assert isinstance(self.warmup_num, int) and self.warmup_num > 0

    
    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)
    
    @property
    def outpath(self):
        if self.outdir:
            return pathlib.Path(self.outdir).joinpath(self.outname)
        return pathlib.Path(__file__).parent.joinpath(self.outname)

    @property
    def train_set(self):
        return self.return_list

    def read(self):
        def locate_and_get_int(line:str, sub:str):
            start_index = line.find(sub)
            if start_index != -1:
                line = line.replace('>',' ')
                line = line.replace(':',' ')
                int_str = line[start_index+len(sub):].split()[0]
                try:
                    int_val = int(int_str)
                except ValueError:
                    int_val = -1 #failed to convert
                return int_val
            return -1

        with open(self.path,"r") as f:
            lines = f.readlines()[3:-4]
            print(f"total lines: {len(lines)}")
            for line in lines:
                #meta
                if "marker" in line:      
                    fid =  locate_and_get_int(line,"function #")
                    if fid != -1:
                        self.func_idx = fid
                #meta
                elif "read" in line or "write" in line:
                    try:
                        parts = line.split()
                        if parts[3] == 'ifetch' or len(parts) < 10:
                            continue
                        address = int(parts[7],16)
                        size = int(parts[4])
                        begin_page_id, end_page_id = get_page_id(address,size)
                        for page_id in range(begin_page_id,end_page_id+1):
                            try:
                                record = {"page_id":page_id,"n_instr":self.ts,"operation":1 if parts[3] == 'read' else 0, "function_id":self.func_idx, "size":size, "address":address,"pc":int(parts[-1],16), "addr_diff":address - self.last_address, "pc_diff":int(parts[-1],16)-self.last_pc}
                            except ValueError:
                                for i in range(len(parts)):
                                    print (f"i: {i}, parts[i]: {parts[i]}")
                            return_val = self.eq.enqueue(record)
                            if return_val is not None:
                                if self.warmup_num <= 0:
                                    x,y = return_val[0]["item"], return_val[1]
                                    excluded_keys = ["page_id", "n_instr"]
                                    features = {key: x[key] for key in x.keys() if key not in excluded_keys}
                                    self.return_list.append((features,y))
                                else:
                                    self.warmup_num -= 1
                            self.record_list.append(record)
                            self.ts += 1
                    except Exception as e:
                        print(f"发生了一个异常: {type(e).__name__} - {e}")
                        for idx in range(len(parts)):
                            print(f"parts[{idx}]: {parts[idx]}")
                        sys.exit(1)
        while len(self.eq.item_dict) != 0:
            return_val = self.eq.dequeue()
            x,y = return_val[0]["item"], return_val[1]
            excluded_keys = ["page_id", "n_instr"]
            features = {key: x[key] for key in x.keys() if key not in excluded_keys}
            self.return_list.append((features,y))

    def output(self):
        if self.outname is None:
            for record in self.record_list:
                print(record)
        with open(self.outpath,"w") as f:
            for l in self.record_list:
                f.write(str(l)+'\n')

if __name__ == "__main__":
    reader = Record_reader("500ktrace.log","/home/zsq/DynamoRIO-Linux-10.0.0/logs")
    reader.read()
    dataset = reader.train_set
    tree = tree.HoeffdingTreeClassifier()
    accu = metrics.Accuracy()
    for x,y in dataset:
        y_pred = tree.predict_one(x)
        tree.learn_one(x,y)
        accu.update(y,y_pred)
    print(accu)

