from mmrotate.models.builder import ROTATED_LOSSES, build_loss
import torch
import pdb

@ROTATED_LOSSES.register_module()
class H2RBoxAFWSLoss(torch.nn.Module):#包括center_loss,circle_loss,order_loss,hb_loss
    def __init__(self, center_loss_cfg, order_loss_cfg, circle_loss_cfg,
                 reduction='mean', loss_weight=1.0):
        super(H2RBoxAFWSLoss, self).__init__()
        self.center_loss = build_loss(center_loss_cfg)
        self.order_loss = build_loss(order_loss_cfg)#实际上就是角度
        self.circle_loss = build_loss(circle_loss_cfg)#CircleIoULoss
        # self.bbox_loss = build_loss(bbox_loss_cfg)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pos_decoded_bbox_preds_aug,
                pos_decoded_target_preds_aug,
                _radius_aug,
                _radius_target,
                pos_angle_preds_aug,
                pos_angle_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted boxes.
            target (torch.Tensor): Corresponding gt boxes.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.

        Returns:
            loss (torch.Tensor)
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        xy_pred = pos_decoded_bbox_preds_aug[..., :2]
        xy_target = pos_decoded_target_preds_aug[..., :2]
        d_a_pred = pos_angle_preds_aug-pos_angle_targets

        center_loss = self.center_loss(xy_pred, xy_target,
                                       weight=weight[:, None],
                                       reduction_override=reduction,
                                       avg_factor=avg_factor)
 
        _radius_aug = torch.clamp(_radius_aug,min=1e-5,max=1e5)
        _radius_target = torch.clamp(_radius_target,min=1e-5,max=1e5)     
        circle_loss = self.circle_loss(
            xy_pred,
            xy_target,
            _radius_aug,
            _radius_target
        )

        order_loss1 = self.order_loss(
            d_a_pred.sin(), torch.zeros_like(d_a_pred), weight=weight,
            reduction_override=reduction, avg_factor=avg_factor)
        order_loss2 = self.order_loss(
            d_a_pred.cos(), torch.zeros_like(d_a_pred), weight=weight,
            reduction_override=reduction, avg_factor=avg_factor)
        loss_bbox = center_loss + torch.min(order_loss1, order_loss2) + 2*circle_loss
        # pdb.set_trace()
        return self.loss_weight * loss_bbox
