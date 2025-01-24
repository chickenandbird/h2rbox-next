from mmrotate.models.builder import ROTATED_LOSSES, build_loss
import torch
import math


@ROTATED_LOSSES.register_module()
class CircleIoULoss(torch.nn.Module):#包括center_loss,circle_loss,order_loss,hb_loss
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(CircleIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                ctr_aug,
                ctr_target,
                _radius_aug,
                _radius_target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Returns:
            loss (torch.Tensor)
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        
        distance = torch.sqrt(torch.sum(torch.square(ctr_aug-ctr_target),dim=-1))

        intersection = torch.zeros_like(distance)
        area_aug = math.pi*torch.square(_radius_aug)
        area_target = math.pi*torch.square(_radius_target)

        #overlap
        condition_overlap = (distance < _radius_aug + _radius_target) & (distance > torch.abs(_radius_aug - _radius_target))
        if condition_overlap.any():
            distance_valid = distance[condition_overlap]
            r1_valid = _radius_aug[condition_overlap]
            r2_valid = _radius_target[condition_overlap]

            part1 = _radius_aug[condition_overlap]**2 * torch.acos((distance_valid ** 2 + r1_valid ** 2 - r2_valid ** 2) / (2 * distance_valid * r1_valid))
            part2 = _radius_target[condition_overlap]**2 * torch.acos((distance_valid ** 2 + r2_valid ** 2 - r1_valid ** 2) / (2 * distance_valid * r2_valid))
            part3 = 0.5 * torch.sqrt((-distance_valid + r1_valid + r2_valid) * (distance_valid + r1_valid - r2_valid) * 
                                        (distance_valid - r1_valid + r2_valid) * (distance_valid + r1_valid + r2_valid))
            
            intersection[condition_overlap] = part1 + part2 - part3

        #contain
        condition_contain = distance <= torch.abs(_radius_aug - _radius_target)
        intersection[condition_contain] = torch.min(area_aug, area_target)[condition_contain]

        union = area_aug + area_target - intersection   
        if reduction== 'sum':     
            return self.loss_weight * (1-torch.sum(intersection/union))
        else:
            return self.loss_weight * (1-torch.mean(intersection/union))



# import torch
# import math
# def circle_iou_loss(radius1, radius2, d):
#     """
#     计算两个圆的 IoU 损失 (1 - IoU)
#     参数：
#         radius1: Tensor，圆1的半径
#         radius2: Tensor，圆2的半径
#         d: Tensor，圆心距离
#     返回：
#         Tensor，IoU 损失
#     """
#     # 圆面积
#     area1 = math.pi * radius1 ** 2
#     area2 = math.pi * radius2 ** 2

#     # 条件1：不相交，IoU = 0
#     # 条件2：一个圆包含另一个圆
#     # 条件3：部分重叠
    
#     # 计算交集面积
#     r1_sq = radius1 ** 2
#     r2_sq = radius2 ** 2
#     intersection = torch.zeros_like(d)
    
#     # 相交情况
#     condition_overlap = (d < radius1 + radius2) & (d > torch.abs(radius1 - radius2))
#     if condition_overlap.any():
#         d_valid = d[condition_overlap]
#         r1_valid = radius1[condition_overlap]
#         r2_valid = radius2[condition_overlap]

#         part1 = r1_sq[condition_overlap] * torch.acos((d_valid ** 2 + r1_valid ** 2 - r2_valid ** 2) / (2 * d_valid * r1_valid))
#         part2 = r2_sq[condition_overlap] * torch.acos((d_valid ** 2 + r2_valid ** 2 - r1_valid ** 2) / (2 * d_valid * r2_valid))
#         part3 = 0.5 * torch.sqrt((-d_valid + r1_valid + r2_valid) * (d_valid + r1_valid - r2_valid) * 
#                                  (d_valid - r1_valid + r2_valid) * (d_valid + r1_valid + r2_valid))
        
#         intersection[condition_overlap] = part1 + part2 - part3

#     # 一个圆包含另一个圆
#     condition_contain = d <= torch.abs(radius1 - radius2)
#     if condition_contain.any():
#         intersection[condition_contain] = torch.min(area1, area2)[condition_contain]

#     # 计算并集面积
#     union = area1 + area2 - intersection

#     # 计算 IoU
#     iou = intersection / union
#     iou_loss = 1 - iou

#     return iou_loss

# # 示例
# radius1 = torch.tensor([3.0, 5.0, 4.0], requires_grad=True)
# radius2 = torch.tensor([2.0, 4.0, 5.0], requires_grad=True)
# d = torch.tensor([1.0, 6.0, 2.0], requires_grad=True)

# loss = circle_iou_loss(radius1, radius2, d)
# print("IoU Loss:", loss)

# # 反向传播
# loss.mean().backward()
# print("Grad of radius1:", radius1.grad)
# print("Grad of radius2:", radius2.grad)
# print("Grad of d:", d.grad)
