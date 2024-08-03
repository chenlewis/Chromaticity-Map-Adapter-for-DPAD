
import os
import random
import json

rootPath = r'E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data\images'

# 图像文件夹路径和类别标签
folder_paths = ["0.Abnormal", "1.Normal"]
folder_labels = [0, 1]
phonedevice = ["MEIZU","ONEPLUS5T","IPHONE6","OPPORENO"]

allimagesFileList = [os.path.join(rootPath,folder_paths[1], i)
                     for i in os.listdir(os.path.join(rootPath,folder_paths[1]))]

allunforgedList = allimagesFileList
print(allunforgedList)
unforgery_deviceset = set(["_".join((i.split("\\")[-1].split(".")[0]).split("_")[:2]) for i in allunforgedList if
                           len((i.split("\\")[-1].split(".")[0]).split("_")) >4])

print(unforgery_deviceset)
allunforgedListReal = []
tempList = []

for device in list(unforgery_deviceset):
    for fileaname in allunforgedList:
        if device in fileaname:
            tempList.append(fileaname)
    random.shuffle(tempList)
    allunforgedListReal = allunforgedListReal + tempList[:int(len(tempList) / 50)]
    tempList = []
allunforgedList = allunforgedListReal

print(len(allunforgedList),allunforgedList)