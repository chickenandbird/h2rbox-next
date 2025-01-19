# import numpy as np

# obboxes = np.array([[0,0,6,8,2*np.pi/3]])

# def obb2hbb_np_le90(obboxes):
#     """Convert oriented bounding boxes to horizontal bounding boxes.

#     Args:
#         obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

#     Returns:
#         hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
#     """
#     # center, w, h, theta = obboxes[..., :2], obboxes[..., 2], obboxes[..., 3], obboxes[..., 4]
#     center, w, h, theta = np.split(obboxes, [2, 3, 4], axis=-1)
#     Cos, Sin = np.cos(theta), np.sin(theta)
#     x_bias = np.abs(w / 2 * Cos) + np.abs(h / 2 * Sin)
#     y_bias = np.abs(w / 2 * Sin) + np.abs(h / 2 * Cos)
#     bias = np.concatenate([x_bias, y_bias], axis=-1)
#     hbboxes = np.concatenate([center - bias, center + bias], axis=-1)
#     _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
#     _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
#     _w = hbboxes[..., 2] - hbboxes[..., 0]
#     _h = hbboxes[..., 3] - hbboxes[..., 1]
#     _theta = np.zeros(theta.shape[0])
#     obboxes1 = np.stack([_x, _y, _w, _h, _theta], axis=-1)
#     obboxes2 = np.stack([_x, _y, _h, _w, _theta - np.pi / 2], axis=-1)
#     obboxes = np.where((_w >= _h)[..., None], obboxes1, obboxes2)
#     return obboxes

# obboxes = obb2hbb_np_le90(obboxes)
# print(len(obboxes[0]))

import torch 
k = torch.randn((4,4,5))
print(torch.max(k,dim=-1))