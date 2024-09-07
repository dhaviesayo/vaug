import numpy as np
from torchvision.io import read_video
from . import functional as f
from . import torch_functional as ff
import torch
from torch.nn import Module

try:
    from decord import VideoReader , cpu
    import decord
    decord.bridge.set_bridge('torch')
except:
    raise Exception("Install decord through 'pip install decord' or 'git clone --recursive https://github.com/dmlc/decord' ")


class speed(Module):
    def __init__(self , factor=1.5 ):
        super(speed,self).__init__()     #defaults to fast mode (1.5),  it can be set to slowed as specified
        self.factor  = factor   #"factor to fasten or slowen video by"
        if factor >= 1:
            self.mode =  "fast"
        else:
            self.mode = "slow"


    def forward(self , video: any ,  vid_fps = None) -> list[torch.Tensor]: #expects batched video BNCHW or list of video paths or list of video tensors in NCHW or one single path
        if type(video) == str:

            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = vr.unsqueeze(0)
            aug = f.change_speed(vr , factor = self.factor ,  mode = self.mode)


        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3 or video.shape[2] == 1
                aug = f.change_speed(video , factor = self.factor ,  mode = self.mode)
                vid_fps = vid_fps
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
            aug = f.change_speed(video, factor = self.factor ,  mode = self.mode)

        else:
            raise Exception(" Expects Video as batched Tensor(NTCHW) , Video Path , list of paths or list of videos")
        
        return aug 

class to_gray(Module):
    def __init__(self, mode : str) :  #mode can be set to all frames or random frames
        super(to_gray,self).__init__()
        if mode == "all" or "random":  
            self.mode =  mode
        else:
            raise Exception ("mode can only be 'all' or 'random' ")
    def forward(self, video : any) -> list[torch.Tensor]:    #expects batched video BNCHW or list of video paths or list of video tensors in NCHW or one single path
        if type(video) == str:
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = vr.unsqueeze(0)
            aug = f.grayscale(vr , mode = self.mode)

        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3;  #expects only 3 channel video
                aug = f.grayscale(video , mode = self.mode)
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
             aug = f.grayscale(video ,  mode = self.mode)


        else:
            raise Exception(" Expects Video in Tensor(NTCHW) or Video Path")
        
        return aug


class resize(Module):
    def __init__(self , size : list[int]):  #size = H,W
        super(resize,self).__init__()
        if type(size) == list and len(size) ==2:
            self.h , self.w = size[0] , size[1]
        else:
            raise Exception("size must be a list containing H and W")

    def forward(self, video: any) -> list[torch.Tensor]:    #expects batched video BNCHW or list of video paths or list of video tensors in NCHW or one single path
        if type(video) == str:
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = vr.unsqueeze(0)
            aug = ff.resize(vr ,[self.h , self.w])

        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3;  #expects only 3 channel video
                aug = ff.resize(video ,[ self.h , self.w])
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
             aug = ff.resize(video , [self.h , self.w])

        else:
            raise Exception(" Expects Video in Tensor(NTCHW) or Video Path")
        
        return aug



class reverse(Module):
    def __init__(self) :
        super(reverse,self).__init__()
        pass
    def forward(self, video : any) -> list[torch.Tensor]:    #expects batched video BNCHW or list of video paths or list of video tensors in NCHW or one single path
        if type(video) == str:
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr[::-1]
            aug = vr.permute(0,3,1,2)

        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3;  #expects only 3 channel video
                aug = f.reverse(video)
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
             aug = f.reverse(video)


        else:
            raise Exception(" Expects Video in Tensor(NTCHW) or Video Path")
        
        return aug 

class perspective(Module): 
    ### torchvision's implementation of random Perspective 
    def __init__(self , distortion_scale = 0.5, fill: list[float] = None):
        super(perspective,self).__init__()
        self.distortion_scale =  distortion_scale
        self.fill = fill

    def forward(self , video: str or torch.Tensor):
        if type(video) == str:
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = vr.unsqueeze(0)
            aug = ff.perspective(vr , distortion_scale = self.distortion_scale , fill = self.fill)

        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3;  #expects only 3 channel video
                aug = ff.perspective(video , distortion_scale = self.distortion_scale , fill = self.fill)
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
            aug = ff.perspective(video , distortion_scale = self.distortion_scale , fill = self.fill)

        else:
            raise Exception(" Expects Video in Tensor(NTCHW) or Video Path")
        
        return aug

class mirror (Module): 
    def __init__(self):
        super(mirror,self).__init__()
    def forward(self , video: str or torch.Tensor):
        if type(video) == str:
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = vr.unsqueeze(0)
            aug = f.mirror(vr)

        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3;  #expects only 3 channel video
                aug = f.mirror(video)
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
            aug = f.mirror(video)

        else:
            raise Exception(" Expects Video in Tensor(NTCHW) or Video Path")
        
        return aug

    

class flip (Module): 
    def __init__(self):
        super(flip,self).__init__()
    def forward(self , video: str or torch.Tensor):
        if type(video) == str:
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = vr.unsqueeze(0)
            aug = f.flip(vr)

        elif type(video) == torch.Tensor:
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3;  #expects only 3 channel video
                aug = f.flip(video)
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:
            aug = f.flip(video)

        else:
            raise Exception(" Expects Video in Tensor(NTCHW) or Video Path")
        
        return aug

        

class add_noise(Module)  : 
    def __init__(self ,  ratio: float):
        super(add_noise,self).__init__()
        self.ratio = ratio
    def forward(self , video: str or torch.Tensor ):
        ratio = self.ratio
        if isinstance(video, (str)):    #file path
            vr = VideoReader(video , cpu(0) ,  num_threads = 3)
            vr = vr.get_batch(range(0, len(vr)))
            vr =  vr.permute(0,3,1,2)
            vr = (vr.unsqueeze(0)/255).to(torch.float32)
            aug = f.add_noise(vr , ratio)

        elif type(video) == torch.Tensor:   # batched tensor
            if len(video.shape) ==  5 :
                assert video.shape[2] == 3 , "expects only 3 channel video tensor"
                aug = f.add_noise(video , ratio)
            else:
                raise Exception ("Expects Video Tensor in 5D (NTCHW)")
        elif type(video) == list and type(video[0]) == torch.Tensor:  #tensors in a list 
            if str in video:
                raise TypeError ("List must contain only video tensors") 
            aug = f.add_noise(video ,  ratio)

        else:
            raise Exception(" Expects Batched Video or Video Path")
        
        return aug




            