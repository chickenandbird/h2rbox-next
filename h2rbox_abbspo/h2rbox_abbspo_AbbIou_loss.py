from mmrotate.models.builder import ROTATED_LOSSES, build_loss
import torch
import math

def obb2xyxy(rbboxes):
    w = rbboxes[:,:,2]
    h = rbboxes[:,:,3]

    dx = rbboxes[:,:, 0]
    dy = rbboxes[:,:, 1]
    x1 = dx - w / 2
    y1 = dy - h / 2
    x2 = dx + w / 2
    y2 = dy + h / 2
    return torch.stack((x1, y1, x2, y2), -1)

def obb2xyxy_v2(rbboxes):
    w = rbboxes[:,2]
    h = rbboxes[:,3]

    dx = rbboxes[:,0]
    dy = rbboxes[:,1]
    x1 = dx - w / 2
    y1 = dy - h / 2
    x2 = dx + w / 2
    y2 = dy + h / 2
    return torch.stack((x1, y1, x2, y2), -1)

def iou_loss(pred_bbox, true_bbox):
    # 计算预测框和真实框的交集
    x1_inter = torch.max(pred_bbox[:,:,0], true_bbox[:,:,0])
    y1_inter = torch.max(pred_bbox[:,:,1], true_bbox[:,:,1])
    x2_inter = torch.min(pred_bbox[:,:,2], true_bbox[:,:,2])
    y2_inter = torch.min(pred_bbox[:,:,3], true_bbox[:,:,3])

    # 计算交集的宽度和高度
    inter_width = torch.clamp(x2_inter - x1_inter, min=0)
    inter_height = torch.clamp(y2_inter - y1_inter, min=0)

    # 计算交集区域的面积
    intersection = inter_width * inter_height
    intersection = intersection.squeeze(-1)
    # 计算预测框和真实框的面积
    pred_area = (pred_bbox[:,:, 2] - pred_bbox[:,:, 0]) * (pred_bbox[:,:, 3] - pred_bbox[:,:, 1]).squeeze(-1)
    true_area = (true_bbox[:,:, 2] - true_bbox[:,:, 0]) * (true_bbox[:,:, 3] - true_bbox[:,:, 1]).squeeze(-1)

    # 计算并集区域的面积
    union = pred_area + true_area - intersection

    # 计算 IoU
    iou = intersection / union

    # IoU Loss = 1 - IoU
    return 1 - iou

@ROTATED_LOSSES.register_module()
class AbbIoULoss(torch.nn.Module):
    def __init__(self, shape_loss_cfg,reduction='mean', loss_weight=1.0):
        super(AbbIoULoss, self).__init__()
        self.shape_loss = build_loss(shape_loss_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                bbox_preds,
                bbox_gt,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Returns:
            loss (torch.Tensor)
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        Ns = 8
        new_bbox_preds = bbox_preds[:,:4]
        angle_preds = bbox_preds[:,4].unsqueeze(1)
        s_w = (1+(1.3-1)*torch.arange(Ns)/Ns)
        s_h = (1+(1.3-1)*torch.arange(Ns)/Ns)
        s1,s2 = torch.meshgrid(s_h,s_w)
        s = torch.stack([s1,s2],dim=-1).view(-1,2).repeat(angle_preds.size(0),1,1).to(new_bbox_preds.device)
        angle_use = angle_preds.unsqueeze(1).repeat(1,Ns*Ns,1).to(new_bbox_preds.device)
        s[angle_preds[:,0].abs()>=math.pi/4,:,:] = ((1-s)*(angle_use.abs())*4/math.pi+1)[angle_preds[:,0].abs()>=math.pi/4,:,:]
        s[angle_preds[:,0].abs()<math.pi/4,:,:] = ((s-1)*(angle_use.abs())*4/math.pi+1)[angle_preds[:,0].abs()<math.pi/4,:,:]
        tensor_one = torch.ones(angle_preds.size(0),Ns*Ns,1,device=new_bbox_preds.device)
        s_use = torch.cat([tensor_one,tensor_one,s],dim=-1).to(new_bbox_preds.device)

        new_bbox_preds_use = new_bbox_preds.unsqueeze(1).repeat(1,Ns*Ns,1)
        bbox_gt_use = bbox_gt[:,:4].unsqueeze(1).repeat(1,Ns*Ns,1)*s_use
        all_iou_loss = iou_loss(obb2xyxy(new_bbox_preds_use),obb2xyxy(bbox_gt_use)).to(new_bbox_preds.device)

        as_loss = weight*torch.min(all_iou_loss,dim=-1)[0]

        if reduction== 'sum':     
            return self.loss_weight *0.01*self.shape_loss(obb2xyxy_v2(new_bbox_preds),obb2xyxy_v2(bbox_gt[:,:4]),weight=weight,avg_factor=avg_factor)+ self.loss_weight *torch.sum(as_loss)
        else:
            return self.loss_weight *0.01*self.shape_loss(obb2xyxy_v2(new_bbox_preds),obb2xyxy_v2(bbox_gt[:,:4]),weight=weight,avg_factor=avg_factor)+ self.loss_weight *torch.mean(as_loss)