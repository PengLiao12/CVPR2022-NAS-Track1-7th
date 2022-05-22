import os


import json
from collections import OrderedDict
import random



####  将channel_sample.txt结果转化为要求提交的json格式
if __name__ == '__main__':


    f = open("./channel_sample.txt", encoding='utf-8')
    res= dict()
    while True:
        line = f.readline()
        if line:
            temp = []
            temp = line.split(" ")
            res[temp[0]]=temp[1]
        else:
            break
    f.close()

    archs = json.load(open('./data/CVPR_2022_NAS_Track1_test.json'))
    if isinstance(archs, dict):
        for key in archs:
            try:
                archs[key]['acc'] = float(res[archs[key]['arch']])
            except:
                break

        json.dump(archs, open('20220517_night_final.json', 'w'))
