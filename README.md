# vaug
A Minimal API for Video preprocessing built upon torch and decord 

`vaug` coined from `video-augmenting`  is a mini API for video preprocessing and its being improved daily. It provides more refined and faster way of preprocessing and its build on top of `decord` an hardware accelerated video decoder.
It handles video path and video tensor batches in `B,N,C,H,W`.

## Installation
```bash
git clone https://github.com/dhaviesayo/vaug
```
d

## Usage

### Perspective

Perspective changes the viewpoint of the video randomly.
```python
from vaug.transforms import perspective
import decord
from decord import VideoReader , cpu
decord.bridge.set_bridge('torch')

perspective = perspective(distortion_scale = 0.5 )
new_view = perspective("/path/to/video.mp4")

or

vr = VideoReader("/path/to/video.mp4" , cpu(0))
vr = vr.get_batch(range(0, len(vr)))  # to get certain amounts of frame
vr =  vr.permute(0,3,1,2)
vr = vr.unsqueeze(0)
new_view = perspective(vr)
```

### Resize

```python
from vaug.transforms import resize
resize = resize([256,512])
resized_video = resize("/path/to/video.mp4") or Tensor
```

### Tile Frames
Return a list containing a single image containing all the frames and x,y coordinates of each tiled frame , only accepts single input
```python
from vaug.utils import tileframes
tiled_frames = tileframes("path/to/video.mp4")
#or
tiled_frames = tileframes(videotensor)
```

### Custom collate function
Handles inconsistent number of frames when using torch's dataloader.
Due to the Stacking process that occurs during batching, inconsistent number of frames would raise an exception

```python
from vaug import collate , collate_label

.....
train_loader = DataLoader(train_set , batch_size = 7 , collate_fn = collate)

#if your dataset contains labels

train_loader = DataLoader(train_set , batch_size , collate_fn = collate_label)

.....rest of your training code

```






