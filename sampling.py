

import random
outList = []
def isswap(array,i,j):#
    if i == j:
        return True
    for n in range(i,j):
        if array[n]!= array[j]:
            continue
        else:
            return False
    return True

#全排列并检查4个stage的上下线
def permutations(arr, begin, end):
    global outList
    if begin == end:
        if (arr[0]>=2) & (arr[0]<=5) & (arr[1]>=2) & (arr[1]<=5) & (arr[2]>=2) & (arr[2]<=8) & (arr[3]>=2) & (arr[3]<=5):
            # print(arr)
            outList.append(list(arr))
    else:
        for index in range(begin, end):
            if isswap(arr,begin,index):
                arr[index], arr[begin]= arr[begin], arr[index]
                permutations(arr, begin +1, end)
                arr[index], arr[begin]= arr[begin], arr[index]

#整数拆分为若干数相加，若若干数为4，进行全排列
def cutNum(cutList):
    if len(cutList) <= 1 or cutList[-1] == 1:
        return
    else:
        last = cutList[-1]
        frontPart = cutList[0:-1]
        for left in range(2, last // 2 + 1):
            if left >= frontPart[-1]:
                right = last - left
                newList = frontPart + [left, right]
                if len(newList) == 4:
                    permutations(newList, 0, 4)
                cutNum(newList)

#在训练阶段，每个epoch按照batch的数量产生对应数量的接受训练的网络架构
#依照51位编码，先按照strem的channel（即第五位编码）的数量分为7部分
#每个部分按照前四位编码，即4个stage的block的数量采样：
### 先生成最少block总数2+2+2+2=8至最大block总数5+5+5+8=23,16个子部分的全排列
### 16个部分按照生成的排序从小到大依次按量采样
### 根据blcok数量随机选择每个block的channel
### 即构成51位编码
def archs_sampling(sampling_num):
    # sampling_num = 2503
    strem = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    stage_block_num = [5, 5, 8, 5]
    frist_group = [0] * len(strem)
    # 第一层划分，根据strem划分采样数量
    first_num = sampling_num // len(strem)
    for i in range(len(strem)):
        if i != (len(strem) - 1):
            frist_group[len(strem) - i - 1] = first_num
        else:
            frist_group[len(strem) - i - 1] = sampling_num - sum(frist_group)

    # 第二层划分，确定每个stage的采样
    archs = []
    for first_i_num in range(len(frist_group)):
        first_i = frist_group[first_i_num]
        second_stage_group = []
        # 首先全排列去重得到不同block在stage的排列
        for second_i in range(8, 24):
            global outList
            outList = []
            for i in range(2, second_i // 2 + 1):
                j = second_i - i
                cutNum([i, j])
            second_stage_group.append(outList)
        # 依照first_i,从outList全排列中依照不同block的数量从小到达选择
        second_stage_group = sorted(second_stage_group, key=len)
        num_second_stage_group = [len(i) for i in second_stage_group]
        ## 按比例抽样first_i个
        plan_num_second_stage_group = [round(i * first_i / sum(num_second_stage_group)) for i in num_second_stage_group]
        if sum(plan_num_second_stage_group) != first_i:
            plan_num_second_stage_group[-1] = plan_num_second_stage_group[-1] + first_i - sum(
                plan_num_second_stage_group)
        ##block抽样
        for count_num in range(len(second_stage_group)):
            plan_res = random.sample(second_stage_group[count_num], plan_num_second_stage_group[count_num])
            for individual_blcok in plan_res:
                individual_blcok_code = [random.randint(1, 7) for i in range(sum(individual_blcok) * 2)]
                individual_code = '1' + ''.join((str(x) for x in individual_blcok)) + str(first_i_num + 1)

                individual_code = individual_code + ''.join(
                    str(x) for x in individual_blcok_code[0:individual_blcok[0] * 2]) + '00' * (
                                              stage_block_num[0] - individual_blcok[0])
                individual_code = individual_code + ''.join(
                    str(x) for x in individual_blcok_code[individual_blcok[0] * 2:sum(plan_res[0][0:2]) * 2]) + '00' * (
                                              stage_block_num[1] - individual_blcok[1])
                individual_code = individual_code + ''.join(
                    str(x) for x in
                    individual_blcok_code[sum(plan_res[0][0:2]) * 2:sum(plan_res[0][0:3]) * 2]) + '00' * (
                                          stage_block_num[2] - individual_blcok[2])
                individual_code = individual_code + ''.join(
                    str(x) for x in individual_blcok_code[sum(plan_res[0][0:3]) * 2:sum(plan_res[0]) * 2]) + '00' * (
                                          stage_block_num[3] - individual_blcok[3])

                archs.append(individual_code)

    return archs


if __name__ == '__main__':
    ars=archs_sampling(2502)
    print(archs_sampling(2502))


