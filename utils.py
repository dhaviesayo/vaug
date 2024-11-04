import numpy as np
from torch.nn import functional as F , Module
from torchvision.transforms import functional as f
import torch

try:
    from decord import VideoReader , cpu , gpu
    import decord
    decord.bridge.set_bridge('torch')
except:
    raise Exception("Install decord through 'git clone --recursive https://github.com/dmlc/decord' ")



def collate(x ,labels=None):  #takes  a list containing video of varied frame rates  , input as #N,C,H,W
    frames_len= []
    for data in x:
        if data.shape[1] > 3:
            raise Exception ("channel is greater than 3 ")
        else:
            f,c,h,w = data.size()
            frames_len.append(f)

    video = []
    for data in x:
        if data.shape[1] > 3:
            raise Exception ("channel is greater than 3 ")
        else:
            f,c,h,w = data.size()

        if f != max(frames_len):
            interpolated = F.interpolate(data.permute(1,2,3,0).type(torch.float32), size= [w , (max(frames_len))],
                          mode = "bilinear" ,  align_corners= True ).permute(3,0,1,2)
            video.append(interpolated)

        else:
            video.append(data)
    batched = torch.stack(video)
    labels_list = []
    if labels is not None:
        for i in labels:
            labels_list.append(i)
        batchedlabels = torch.stack(labels_list)
        return batched  , batchedlabels 
    else:
        return batched   #returns as B,N,C,H,W



def collate_label(x):  #takes a list containing videos of varied frame rates  and corresponding labels  #N,C,H,W
    frames_len= []
    for data in x:
        if isinstance(data , tuple):
            f,c,h,w = data[0].size()
            frames_len.append(f)

    video = []
    labels = []
    for data in x:
        f,c,h,w = data[0].size()
        if f != max(frames_len):
            vid = data[0].permute(1,2,3,0)
            interpolated = F.interpolate(vid.type(torch.float32), size=[w, (max(frames_len))],
                          mode = "bilinear" ,  align_corners= True ).permute(3,0,1,2)
            video.append(interpolated)

        else:
            video.append(data[0])

        l = data[1]
        labels.append(torch.as_tensor(l) if type(l) != torch.Tensor else l)

    batched = torch.stack(video)

    batchedlabels = torch.stack(labels)
    return batched  , batchedlabels #returns as B,N,C,H,W



class sequential(Module):    #takes list of mulitiple tramsforms
    def __init__(self , params):
        super(sequential , self).__init__() 
        self.params = params


    def forward(self,  x):   
        params = self.params
        for transform in params:
            x = transform(x)
        
        return x 



def factors(num):
    factors = []
    for i in range(1,num+1):
        if num%i==0:
            factors.append(i)
        else:
            pass
    return factors


def chunk (vid ,  factors  =  factors):
    f,c,h,w =  vid.shape[0] ,vid.shape[1]  ,vid.shape[2] , vid.shape[3]
    ff = factors(f)
    if len(ff) == 2 :
        f = f-1
        ff = factors(f)
        chunk_size = ff[int(int(len(ff)-1) / 2)]
        frame_size = int(f / chunk_size)
        
        
    else:
        chunk_size = ff[int(int(len(ff)-1) / 2)]
        frame_size = int(f / chunk_size)
    frames = []   
    for frame in vid:
        frames.append(torch.nn.MaxPool2d(3)(frame.type(torch.float32)))
    del vid

    chunked_frames =[]
    for i in range(chunk_size) :
        chunked = frames[:frame_size]
        chunked_frames.append(chunked)
        del frames[:frame_size]

    if len(frames)==1:
        chunked_frames[-1].extend( frames)
        del frames
        return chunked_frames
    else:
        del frames
        return chunked_frames  # ex vid of 6 frames : [[frame1,frame2,frame3] , [frame4,frame5,frame6]]



def chunkframes(vid: str or torch.Tensor,  chunk =chunk , factors=factors) -> list[torch.Tensor] :
    if type(vid) == str:
        vid = VideoReader(vid , cpu(0) , num_threads =2)
        vid = vid.get_batch(range(0, len(vid)))
        vid =  vid.permute(0,3,1,2)
    elif type(vid) == torch.Tensor:
         vid = vid
    else:
        raise ValueError("Expects Video path or Video tensor TCHW ")
    chunked = chunk(vid , factors)
    tiles = []
    for b in chunked:
        ff = factors(len(b))
        if len(factors(len(b))) ==2:
            b_x = torch.stack(b)  #4D
            b = b_x.reshape(b_x.shape[0] , b_x.shape[1] , b_x.shape[2]*b_x.shape[3])
                                                                                       ##interpolate b by adding extra 1 frame
            b_interp = F.interpolate(b.permute(1,2,0) , size= b_x.shape[0]+1, mode="nearest" )
            new_b =  b_interp.permute(2,0,1).reshape(b_x.shape[0]+1,b_x.shape[1],b_x.shape[2],b_x.shape[3] )
            b = [frames  for frames in new_b]
            ff = factors(len(b))
            mid  = int(int(len(ff) - 1 )/ 2)
            h_w = [ff[mid] , int(len(b) / ff[mid])]
            w = max(h_w)
            h = min(h_w)


            tobeconcat = []
            for i in range(int(h)) :
                totile = b[:w]
                concat = torch.cat(totile , dim=2)
                tobeconcat.append(concat)
                del b[:w]
            out = torch.cat(tobeconcat ,  dim=1)
        else:
            ff = factors(len(b))
            mid  = int(int(len(ff) - 1 )/ 2)
            h_w = [ff[mid] , int(len(b) / ff[mid])]
            w = max(h_w)
            h = min(h_w)


            tobeconcat = []
            for i in range(int(h)) :
                totile = b[:w]
                concat = torch.cat(totile , dim=2)
                tobeconcat.append(concat)
                del b[:w]
            out = torch.cat(tobeconcat ,  dim=1)
        tiles.append(out)
    del tobeconcat
    del totile
    del concat
    del out
    
    tiles = [tiles/255 for tiles in tiles]
    return tiles #[tile1 , tile2 ,  tile3 ....]



def _chunk (vid ):
    f,c,h,w =  vid.shape[0] ,vid.shape[1]  ,vid.shape[2] , vid.shape[3]
    
    chunk_size = 1
    frame_size = int(f / chunk_size)
    frames = []   
    for frame in vid:
        frames.append(torch.nn.MaxPool2d(3)(frame.type(torch.float32)))
    del vid

    chunked_frames =[]
    for i in range(chunk_size) :
        chunked = frames[:frame_size]
        chunked_frames.append(chunked)
        del frames[:frame_size]

    if len(frames)==1:
        chunked_frames[-1].extend( frames)
        del frames
        return chunked_frames
    else:
        del frames
        return chunked_frames  # ex vid of 6 frames : [[frame1,frame2,frame3] , [frame4,frame5,frame6]]



def tileframes(vid: list or torch.Tensor,  chunk = _chunk , factors=factors , height =  256 , width = 512) -> list[torch.Tensor] :  # outputs tild image and coordinated of each frame in the tiled image
    if isinstance(vid , list):  #lists of path
        device = gpu(0) if torch.cuda.is_available() else cpu(0)
        vid   =  [VideoReader(paths, device , num_threads =2) for paths in vid]
        vid = [vid.get_batch(range(0, len(vid))).permute(0,3,1,2) for vid in vid]

    elif type(vid) == torch.Tensor:   #batched tensor
         vid = vid
    else:
        raise ValueError("Expects list of Video paths or batched Video tensor  as BTCHW ")
    tiles = []
    for vid in vid:
        chunked = chunk(vid)
        for b in chunked:
            ff = factors(len(b))
            mid  = int(int(len(ff) - 1 )/ 2)
            h_w = [ff[mid] , int(len(b) / ff[mid])]
            w = max(h_w)
            h = min(h_w)

            b_x = torch.stack(b)  #4D
            _b = b_x.reshape(b_x.shape[0] , b_x.shape[1] , b_x.shape[2]*b_x.shape[3])
            interpsize = b_x.shape[0]+ 1 
            if len(factors(len(b))) ==2 or abs(w-h) > 12:
                while len(factors(len(b))) < 3  or  abs(w-h) > 12:                                                                      ##interpolate b by adding extra 1 frame
                    b_interp = F.interpolate(_b.permute(1,2,0) , size= interpsize, mode="nearest" )
                    new_b =  b_interp.permute(2,0,1).reshape(interpsize,b_x.shape[1],b_x.shape[2],b_x.shape[3] )
                    b = [frames  for frames in new_b]
                    ff = factors(len(b))
                    mid  = int(int(len(ff) - 1 )/ 2)
                    h_w = [ff[mid] , int(len(b) / ff[mid])]
                    w = max(h_w)
                    h = min(h_w)
                    interpsize +=1

                tobeconcat = []
                for i in range(int(h)) :
                    totile = b[:w]
                    concat = torch.cat(totile , dim=2)
                    tobeconcat.append(concat)
                    del b[:w]
                out = torch.cat(tobeconcat ,  dim=1)

            else:
                interpsize = len(b)
                ff = factors(len(b))
                mid  = int(int(len(ff) - 1 )/ 2)
                h_w = [ff[mid] , int(len(b) / ff[mid])]
                w = max(h_w)
                h = min(h_w)


                tobeconcat = []
                for i in range(int(h)) :
                    totile = b[:w]
                    concat = torch.cat(totile , dim=2)
                    tobeconcat.append(concat)
                    del b[:w]
                out = torch.cat(tobeconcat ,  dim=1)
            tiles.append(out)
        del totile
        del concat
        del out
        
    tiles = [f.resize(tiles/255 ,  [height , width]) for tiles in tiles]
    '''
    num_frames_in_height = len(tobeconcat)
    num_frames_in_width  = interpsize//  num_frames_in_height
    
    width_diff = tiles[0].shape[-1] // num_frames_in_width
    height_diff = tiles[0].shape[-2] // num_frames_in_height
    w_indices = [i for i in range(0,tiles[0].shape[-1] + width_diff, width_diff)]
    h_indices = [i for i in range(0,tiles[0].shape[-2] + height_diff, height_diff)]
    xy_coor = []
    for i in range(len(h_indices)-1):
        y_coor = h_indices[i:i+2]
        for j in range(len(w_indices)-1):
            x_coor = w_indices[j:j+2]
            coor = x_coor + y_coor
            coor = [coor[0],coor[2],coor[1],coor[3]]    #x1,y1,x2,y2
            xy_coor.append(torch.tensor(coor) )
    '''
    return tiles if len(tiles) ==1 else torch.stack(tiles)


            
        
