# 先预先读取并整理整个trace的访问记录
# 在运行时按时间窗口跟踪
# list of dict
import pathlib
import numpy as np
from collections import OrderedDict
from typing import Optional
from sortedcontainers import SortedList
from util import *

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
    def __init__(self, max_size,hot_thred = None,func_hot_judge = None):
        self.max_size = max_size
        self.item_dict = OrderedDict()
        self.ts = 0
        self.func_hot_judge = func_hot_judge if func_hot_judge is not None else default_hot_gunction #热度维护函数
        
        self.hot_thred = hot_thred
        self.hot_list = SortedList()

    def __str__(self):
        return self.item_dict.__str__()

    @property
    def p80hot(self):
        return self.hot_list[int(0.8*len(self.hot_list))]

    @property
    def MAX_TS(self):
        return 1,000,000,000

    def enqueue(self, item) -> Optional[dict]:
        item_key = item["page_id"]
        return_val = None
        self.ts = item["n_instr"]
        if item_key in self.item_dict:
            self.item_dict[item_key]["access"] += 1
            self.item_dict[item_key]["last_access"] = self.ts
            # if self.func_hot_judge(self.item_dict[item_key],self.hot_thred):
            #     return_val = {"page_id":item_key,"statistics":self.item_dict[item_key], "isHot":1}
            #     self.item_dict.pop(item_key)
            return return_val
        elif len(self.item_dict) == self.max_size:
            return_val = self.dequeue()
        self.item_dict[item_key] = {"access":1,"enqueue_instr":item["n_instr"],"last_access":self.ts,"item":item}
        return return_val

    def dequeue(self) -> Optional[dict]:
        kv = self.item_dict.popitem(last=False)
        kv[1]["last_access"] = self.ts
        if len(self.hot_list > 0):
            self.hot_thred = self.p80hot
        isHot = self.func_hot_judge(kv[1],self.hot_thred)
        item = kv[1]
        self.hot_list.add(item["access"] / (item["last_access"]-item["enqueue_instr"]))
        # self.hot_thred = self.hot_list[int(0.8*len(self.hot_list))]
        return (kv[1], isHot)

    def get_item_by_id(self, item_id):
        return self.item_dict.get(item_id)


class Record_reader:
    def __init__(self, name:str=None, dir:str = None,eq_capacity:int = 100, hot_thred:float = 0.134, outname:str = None, outdir:str = None, warmup_num: int = 1000) -> None:
        self.filename = name
        self.directory = dir
        self.thread_idx = -1
        self.func_idx = -1
        self.outdir = outdir
        self.outname = outname
        self.record_list = []
        self.eq = Evaluation_queue(eq_capacity,hot_thred=hot_thred)
        self.return_list = []
        self.warmup_num = warmup_num
        self.last_pc = 0
        self.last_address = 0
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
        assert False
        with open(self.path,"r") as f:
            lines = f.readlines()[3:-4]
            for line in lines:
                #meta
                if "marker" in line:      
                    fid =  locate_and_get_int(line,"function #")
                    if fid != -1:
                        self.func_idx = fid
                #meta
                elif "read" in line or "write" in line:
                    parts = line.split()
                    if parts[3] == 'ifetch':
                        continue
                    address = int(parts[7],16)
                    size = int(parts[4])
                    begin_page_id, end_page_id = get_page_id(address,size)
                    n_instr = len(self.record_list)
                    for page_id in range(begin_page_id,end_page_id+1):
                        record = {"page_id":page_id,"n_instr":n_instr,"operation":1 if parts[3] == 'read' else 0, "function_id":self.func_idx, "size":size, "address":address,"pc":int(parts[-1],16), "addr_diff":address - self.last_address, "pc_diff":int(parts[-1],16)-self.last_pc}
                        return_val = self.eq.enqueue(record)
                        if return_val is not None:
                            if self.warm_num <= 0:
                                x,y = return_val[0]["item"], return_val[1]
                                excluded_keys = ["page_id", "n_instr"]
                                features = {key: x[key] for key in x.keys() if key not in excluded_keys}
                                self.return_list.append((features,y))
                            else:
                                self.warm_num -= 1
                        self.record_list.append(record)
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

