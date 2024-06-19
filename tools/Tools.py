import torch
# from pre_config import *
from copy import deepcopy
import os
from pathlib import Path
class Tools:
    def __init__(self):
        pass

    def colorstr(self,*input):
        # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
        *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
        colors = {
            'black': '\033[30m',  # basic colors
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',  # 洋红
            'cyan': '\033[36m',  # 青
            'white': '\033[37m',
            'bright_black': '\033[90m',  # bright colors
            'bright_red': '\033[91m',
            'bright_green': '\033[92m',
            'bright_yellow': '\033[93m',
            'bright_blue': '\033[94m',
            'bright_magenta': '\033[95m',
            'bright_cyan': '\033[96m',
            'bright_white': '\033[97m',
            'end': '\033[0m',  # 颜色设置结束标志
            'bold': '\033[1m',  # 加粗
            'underline': '\033[4m'}  # 下划线
        return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

    # def model_info(self, model, verbose=False, imgsz=640):
    #     # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    #     n_p = sum(x.numel() for x in model.parameters())  # number parameters
    #     n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    #     if verbose:
    #         print(
    #             f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
    #         for i, (name, p) in enumerate(model.named_parameters()):
    #             name = name.replace('module_list.', '')
    #             print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
    #                   (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
	#
    #     try:  # FLOPs
    #         try:
    #             import thop  # for FLOPs computation
    #         except ImportError:
    #             thop = None
    #         p = next(model.parameters())
    #         stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32  # max stride
    #         im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
    #         flops = thop.profile(deepcopy(model), inputs=(im,), verbose=False)[0] / 1E9 * 2  # stride GFLOPs
    #         imgsz = imgsz if isinstance(imgsz, list) else [imgsz, imgsz]  # expand if int/float
    #         fs = f', {flops * imgsz[0] / stride * imgsz[1] / stride:.1f} GFLOPs'  # 640x640 GFLOPs
    #     except Exception:
    #         fs = ''
	#
    #     name = Path(model.yaml_file).stem.replace('yolov5', 'YOLOv5') if hasattr(model, 'yaml_file') else 'Model'
    #     Logger.Logger.info(f"{name} summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")

    def ModelStructure(self,path,root,imgsz = 640, ModelStureFlag=True,device = 'cuda:0'):
        if ModelStureFlag:
            from models.yolo import Model
            path = Path(path)
            cfg = root+'/models/'+path.stem+'.yaml'
            model = Model(cfg).to("cuda:0")
            x = torch.randn(1, 3, imgsz, imgsz).to(device)
            script_model = torch.jit.trace(model, x, strict=False)
            script_model.save(root+'/temp/'+path.stem+'_Structure.pt')
        return ModelStureFlag

    def MkDir(self,project,name):
        import glob
        i = 1
        ProjectPath = Path(project)
        ProjectPath.mkdir(parents=True,exist_ok=True)
        # if not os.path.exists(project):
        #     os.mkdir(project)
        #     import pre_config
        #     pre_config.Logger.Logger.info(f'{project} does not exixt...')
        #     pre_config.Logger.Logger.info(f'Creating {project} complete.')
        ProjectFileList = glob.glob(project+'/*')
        ProjectFilePathList = [Path(x).name for x in ProjectFileList]
        while name + str(i) in ProjectFilePathList:
            if len(glob.glob(f'{project}/{name}{str(i)}/*.*'))==0:
                return Path(f'{project}/{name}{str(i)}')
            i+=1
        Pathtemp = Path(f'{project}/{name}{str(i)}')
        Pathtemp.mkdir(parents=True,exist_ok=True)
        return Pathtemp

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

