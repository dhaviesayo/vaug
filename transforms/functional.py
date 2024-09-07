import numpy as np
from torchvision.io import read_video
from torch.nn import functional as F
import torch
try:
    from decord import VideoReader , cpu
    import decord
    decord.bridge.set_bridge('torch')
except:
    raise Exception("Install decord through 'pip install decord' or 'git clone --recursive https://github.com/dmlc/decord' ")



def change_speed(batched_video, factor, mode ):   #input B,N,C,H,W  or  list containing video data
    aug_list = []
    for video_tensor in batched_video:   #1,4,3,234,234
        factor =  np.clip(factor , 0.1 , 2.5)
        if mode == 'fast':
            assert factor > 1
            num_frames = video_tensor.shape[0]
            step_frames = list(np.arange(0 , num_frames , factor).astype(int))
            augmented = video_tensor[[step_frames]]
            aug_list.append(augmented)
        elif mode == "slow":
            assert factor < 1;
            num_frames = video_tensor.shape[0]
            new_num_frames =  int(num_frames * (1/factor))
            flattened_vid = []
            for frames in video_tensor:
                frames_reshape = frames.reshape(3 , frames.shape[1] * frames.shape[2])
                flattened_vid.append(frames_reshape)

            flattened_vid =  torch.stack(flattened_vid)

            interpolated = F.interpolate(flattened_vid.permute(1,2,0).type(torch.float64), size=(new_num_frames),
                          mode = "linear" ,  align_corners= True )

            augmented = interpolated.permute(2,0,1).reshape(new_num_frames,video_tensor.shape[1], video_tensor.shape[2] , video_tensor.shape[3])
            aug_list.append(augmented)
        else:
            raise ValueError("mode must be either 'fast' or 'slow'. ")
        

    return aug_list    #returns a list containing individual tranformed video in the format N,C,H,W

def grayscale(videos ,  mode ):   #input B,N,C,H,W or  list containing videos
    videos_comp = []
    if mode == "all":
        for video in videos:
            gray_frames = []
            f , c ,h ,w  =  video.shape[0] , video.shape[1] , video.shape[2] , video.shape[3] #expects input in N,C,H,W
            for frame in video:
                r ,  g ,  b = frame  #expects a 3-channel image 
                gray = (0.21*r) + (0.72*g) + (0.07*b)
                x3_gray = torch.stack([gray , gray , gray])  #duplicate the grayscale frame to a 3channel frame
                gray_frames.append(x3_gray)
            stacked_gray = torch.stack(gray_frames)
            videos_comp.append(stacked_gray)
    elif mode == "random":
        frame_idx=-1 
        for video in videos:
            gray_frames = []
            f , c ,h ,w  =  video.shape[0] , video.shape[1] , video.shape[2] , video.shape[3]
            rand_indices = np.random.choice(f, int(0.4*f) , replace = False)  #randomly select 40% of the the total frames in the video
            for frame in video:
                frame_idx +=1
                if frame_idx in rand_indices:
                    r ,  g ,  b = frame
                    gray = (0.21*r) + (0.72*g) + (0.07*b)
                    x3_gray = torch.stack([gray , gray , gray])  
                    gray_frames.append(x3_gray)
                else:
                    gray_frames.append(frame)
            gray_stacked = torch.stack(gray_frames)
            videos_comp.append(gray_stacked)
    else:
        raise ValueError("Mode must be 'all' or 'random'.")
        
    return videos_comp    #returns a list containing individual tranformed video in the format N,C,H,W


def reverse(videobatch):
    comp =[]
    for video in videobatch:
        video = video.numpy()
        video = np.flipud(video)
        comp.append(video)
    
    return comp


def mirror(videobatch):
    mirr =  []
    for video in videobatch:
        vid_mirror =  torch.flipud(torch.flip(video , dims=(0,3)))
        mirr.append(vid_mirror / 255)

    return mirr


def flip(videobatch):
    flipped =  []
    for video in videobatch:
        vid_flip =  torch.flipud(torch.flip(video , (0,2)))
        flipped.append(vid_flip / 255)

    return flipped


def add_noise(videobatch , ratio):  #max ratio = 100
    noised = []
    assert ratio < 21 and ratio >= 0.9  , "ratio must between 0 -  20 " ; 
    for vid in videobatch:
        rand_float = torch.rand_like(vid) ** (20 - ratio)
        vid_noise = vid + rand_float
        noised.append(vid_noise)
    return noised