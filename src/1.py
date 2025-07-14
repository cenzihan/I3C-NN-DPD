# 现有一个服务器集群（服务器数量为 serverNum），和一批不同类型的任务（用数组 task表示，下标表示任务类型，值为任务数量）。
# 现需要把这批任务都分配到集群的服务器上，分配规则如下：
# •	应业务安全要求，不同类型的任务不能分配到同一台服务器上
# •	一种类型的多个任务可以分配在多台服务器上
# 「负载」定义为某台服务器所分配的任务个数，无任务的服务器负载为0。
# 「最高负载」定义为所有服务器中负载的最大值。
# 请你制定分配方案，使得分配后「最高负载」的值最小，并返回该最小值。
# 解答要求时间限制：1000ms, 内存限制：256MB
# 输入
# 第一行一个整数，表示集群中服务器的数量 serverNum，1 ≤ serverNum ≤ 10^9
# 第二行一个整数，表示这批任务有 taskTypeNum 种类型，1 ≤ taskTypeNum ≤ 100000，且 taskTypeNum <= serverNum
# 第三行输入 taskTypeNum 个整数，为 task 数组，表示这批任务，1 ≤ task[i] ≤ 10^9
# 输出
# 一个整数，表示「最高负载」的最小值
# 样例
# 输入样例 1 复制
# 5
# 2
# 7 4
# 输出样例 1
# 3
# 提示样例 1
# 类型 0 的任务有 7 个，可表示为 0000000；类型 1 的任务有 4 个，可表示为 1111 ：
# •	按 11、11、00、00、000 或 1、111、00、00、000 分配给 5 台服务器，该分配方案的「最高负载」值为 3，是最小的
# •	其它方案的「最高负载」值都更大，例如 11、11、0000、000 的「最高负载」为 4
# 说明：
# 任务0和任务1不能分配到同一台服务器上。
# 一次性制定分配方案，不存在二次分配。
#
# 输入样例 2 复制
# 8
# 5
# 101 1 1 20 40
# 输出样例 2
# 34
# 提示样例 2
# •	如果「最高负载」为 1，需要的服务器台数为 101 + 1 + 1 + 20 + 40 = 163，超过了给定的服务器数量。因此需要尝试更大的「最高负载」值
# …
# •	如果「最高负载」为 33， 需要的服务器台数为 9
# •	如果「最高负载」为 34， 需要的服务器台数为 8
# •	如果「最高负载」为 35， 需要的服务器台数也为 8
# …
# •	如果「最高负载」为 40， 需要的服务器台数为 7，服务器有浪费
# …
# •	如果「最高负载」为 101， 需要的服务器台数为 5，服务器有更多浪费
# 所以「最高负载」值最小为 34

import math
def can_distribute(tasks, max_load, serverNum):
    """
    Check if it's possible to distribute the tasks among the servers such that
    no server has a load more than max_load.
    """
    total_servers_needed = 0
    for task in tasks:
        # Calculate the number of servers needed for this task type
        total_servers_needed += math.ceil(task / max_load)

    return total_servers_needed <= serverNum

def find_min_max_load(serverNum, tasks):
    """
    Find the minimum possible maximum load among all the servers.
    """
    left, right = 1, max(tasks)  # The possible range for max load

    while left < right:
        mid = (left + right) // 2

        if can_distribute(tasks, mid, serverNum):
            right = mid
        else:
            left = mid + 1

    return left

# Test the function with the provided examples
# example1 = find_min_max_load(5, [7, 4])
example2 = find_min_max_load(8, [101, 1, 1, 20, 40])

# print(example1)
print(example2)

# -----------------------------------------
# import numpy as np
# import math as math
#
# def minimize_max_load(serverNum, taskTypeNum, tasks):
#     min_maxload = 1
#
#     def can_complete(min_maxload, serverNum, taskTypeNum, tasks):
#         severs = 0
#         loads = 0
#         maxtaskNum = [0]*taskTypeNum
#
#         for i in range(taskTypeNum):
#             severs = math.ceil(tasks[i] / min_maxload)
#             maxtaskNum[i] = tasks[i] % serverNum
#             if severs > serverNum:
#                 return False
#
#             loads += math.floor(tasks[i] / serverNum)
#
#
#         if sum(maxtaskNum) > serverNum:
#             loads += sum(maxtaskNum) - serverNum
#             if loads > min_maxload:
#                 return False
#         else:
#             loads += 1
#             if loads > min_maxload:
#                 return False
#
#     while not can_complete(min_maxload, serverNum, taskTypeNum, tasks):
#             min_maxload += 1
#
#     return min_maxload
#
# # serverNum = int(input())
# # taskTypeNum = int(input())
# # tasks = list(map(int,input().split()))
# min_maxload = minimize_max_load(8, 5, [101,1,1,20,40])
# # min_maxload = minimize_max_load(serverNum, taskTypeNum, tasks)
# print(min_maxload)