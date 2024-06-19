import torch
import time
import contextlib

class Timer(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    #上下文管理器，用于计算时间的，包括在CPU和GPU中运行的时间
    # 在进入 with 块时，它记录当前时间；在退出 with 块时，它再次记录当前时间，计算两次记录之间的时间差，并将时间差累加到 t 属性上。
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()