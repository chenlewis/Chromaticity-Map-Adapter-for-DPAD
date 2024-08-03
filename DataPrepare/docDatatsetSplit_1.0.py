
import os
import random
import json

rootPath = r'E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data\images'

# 图像文件夹路径和类别标签
folder_paths = ["0.Abnormal", "1.Normal"]
folder_labels = [0, 1]
phonedevice = ["MEIZU","ONEPLUS5T","IPHONE6","OPPORENO","HUAWEIP9"]


allimagesFileList = [os.path.join(rootPath,folder_paths[1], i)
                     for i in os.listdir(os.path.join(rootPath,folder_paths[1]))]

allunforgedList = allimagesFileList
unforgery_deviceset = set(["_".join((i.split("\\")[-1].split(".")[0]).split("_")[:2]) for i in allunforgedList if
                           len((i.split("\\")[-1].split(".")[0]).split("_")) >4])

print(unforgery_deviceset)
allunforgedListReal = []
tempList = []

for device in list(unforgery_deviceset):
    if device.split("_")[1] in phonedevice:
        continue
    for fileaname in allunforgedList:
        if device in fileaname:
            tempList.append(fileaname)
    random.shuffle(tempList)
    allunforgedListReal = allunforgedListReal + tempList[:int(len(tempList) / 30)]
    tempList = []
allunforgedList = allunforgedListReal



forgeryimagesFileList = [os.path.join(rootPath,folder_paths[0], i)
                     for i in os.listdir(os.path.join(rootPath,folder_paths[1]))]

allforgedList = forgeryimagesFileList
forgery_deviceset = set(["_".join((i.split("\\")[-1].split(".")[0]).split("_")[:2]) for i in forgeryimagesFileList if
                           len((i.split("\\")[-1].split(".")[0]).split("_")) >4])

print(len(allunforgedList))
print(forgery_deviceset)

# 读取文件夹中的图像路径
image_data1 = {}
image_data0 = {}
for folder_path, folder_label in zip(folder_paths, folder_labels):
    for filename in os.listdir(os.path.join(rootPath,folder_path)):

        if len(filename.split("_")) > 4:
            if filename.endswith(".png") and folder_label == 0 and not(filename.split("_")[1] in phonedevice):
                print(filename.split("_")[1])
                if filename.split("_")[1] in phonedevice:
                    print(filename.split("_")[1])
                image_path = os.path.join(folder_path, filename)
                image_label = folder_label
                image_data0[image_path] = image_label
        else:
            if filename.endswith(".png") and folder_label == 0:
                image_path = os.path.join(folder_path, filename)
                image_label = folder_label
                image_data0[image_path] = image_label

for imagefileName in allunforgedList:
    image_data1["\\".join(imagefileName.split("\\")[-2:])] = 1

# 将图像路径和标签打包成元组列表并随机打乱
data1 = list(image_data1.items())
data0 = list(image_data0.items())
print(len(data1), len(data0))
random.shuffle(data1)
random.shuffle(data0)

trainPicList = []
validPicList = []
testPicList = []
tempList = []
for device in list(unforgery_deviceset):
    for (fileaname,label) in data1:
        if device in fileaname:
            tempList.append((fileaname,1))
    random.shuffle(tempList)
    trainPicList = trainPicList + tempList[:int(len(tempList) * 0.8)]
    validPicList = validPicList + tempList[int(len(tempList) * 0.8):int(len(tempList) * 0.9)]
    testPicList = testPicList + tempList[int(len(tempList) * 0.9):int(len(tempList))]
    tempList = []

for device in list(forgery_deviceset):
    for (fileaname, label) in data0:
        if device in fileaname:
            tempList.append((fileaname,0))
    random.shuffle(tempList)
    trainPicList = trainPicList + tempList[:int(len(tempList) * 0.8)]
    validPicList = validPicList + tempList[int(len(tempList) * 0.8):int(len(tempList) * 0.9)]
    testPicList = testPicList + tempList[int(len(tempList) * 0.9):int(len(tempList))]
    tempList = []



random.shuffle(trainPicList)
random.shuffle(validPicList)
random.shuffle(testPicList)

train_dataDict = dict(trainPicList)
val_dataDict = dict(validPicList)
test_dataDict = dict(testPicList)

print(len(list(train_dataDict.keys())), len(list(val_dataDict.keys())), len(list(test_dataDict.keys())))

# 将字典保存为json文件
with open("train.json", "w") as f:
    json.dump(train_dataDict, f)
with open("val.json", "w") as f:
    json.dump(val_dataDict, f)
with open("test.json", "w") as f:
    json.dump(test_dataDict, f)


