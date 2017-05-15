#!/usr/bin.env python
# -*- coding:utf-8 -*-
'''归并排序'''


def MergeSort(lists):
    if len(lists) <= 1:
        return lists
    num = int(len(lists) / 2)
    left = MergeSort(lists[:num])
    right = MergeSort(lists[num:])
    return Merge(left, right)


def Merge(left, right):
    r, l = 0, 0
    result = []
    while l < len(left) and r < len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l += 1
        else:
            result.append(right[r])
            r += 1
    result += right[r:]
    result += left[l:]
    return result

print MergeSort([1, 3, 422, 21, 2, 55, 232, 111, 57, 65])
