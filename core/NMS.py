import torch

# from pre_config import *
import numpy as np
import torchvision


class NMS:
    def __init__(self):
        pass

    def NMSFunc(self,pred,conf_thres,iou_thres,max_det = 300):
        assert  0<=conf_thres<=1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(pred,(list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            pred = pred[0]  # select only inference output\
        device = pred.device
        BatchSize = pred.shape[0]
        ConfPred = pred[...,4]>conf_thres
        MaxBoxWH = 7680
        MaxBoxNum = 30000
        output = [torch.zeros((0,6),device = device)]*BatchSize
        for num,x in enumerate(pred):
            x = x[ConfPred[num]]
            if not x.shape[0]:
                continue
            x[:,5:]*=x[:,4:5]   # 取第四列得到一个列向量，x[:,4]得到一个横向量
                                # conf * scores
            box = self.xywh2xyxy(x[:,:4])
            # 取 x 张量中每一行的从第 5 列到第 85 列的数据，并找到每行的最大值，结果是一个张量，维度为 (batch_size, 1)，其中 batch_size 是 x 张量的第 1 维的大小。
            # keepdim=True 参数表示保持计算结果的维度不变，即结果张量的维度仍为 (batch_size, 1)，而不是被压缩成 (batch_size,)。
            # 找最合适的标签，返回的j是索引，并且是不是x中的索引
            conf, j = x[:,5:pred.shape[2]].max(1,keepdim=True)
            x = torch.cat((box,conf,j.float()),1)[conf.view(-1)>conf_thres]  # 拼接，然后取conf * scores大于conf_thres的
            NumBox = x.shape[0]
            if not NumBox:
                continue

            # 第五列的值降序排列
            x = x[x[:,4].argsort(descending=True)[:MaxBoxNum]]
            c = x[:,5:6] * MaxBoxWH
            # 这个C是偏移量，用来进行不同类的独立NMS，防止各种图堆在一起，影响NMS的结果
            # 但是这里的偏移量好像到了最后也没有处理，直接输出出去了
            boxes, scores = x[:,:4]+c,x[:,4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            i = i[:max_det]
            output[num] = x[i]

        return output



    def xywh2xyxy(self,x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
