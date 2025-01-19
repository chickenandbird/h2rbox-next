from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead, INF

regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                (512, INF))
norm_on_bbox = True
strides=[8, 16, 32, 64, 128]
num_classes = 15
center_sampling = True
center_sample_radius = 1.5
import torch
from functools import partial
def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def get_targets(points, gt_bboxes_list, gt_labels_list):
    assert len(points) == len(regress_ranges)
    num_levels = len(points)
    # expand regress ranges to align with points
    expanded_regress_ranges = [
        points[i].new_tensor(regress_ranges[i])[None].expand_as(
            points[i]) for i in range(num_levels)
    ]
    # concat all levels points and regress ranges
    concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
    concat_points = torch.cat(points, dim=0)

    # the number of points per img, per lvl
    num_points = [center.size(0) for center in points]

    # get labels and bbox_targets of each image
    (labels_list, bbox_targets_list, angle_targets_list,
        _gt_idx_list) = multi_apply(
        _get_target_single,
        gt_bboxes_list,
        gt_labels_list,
        points=concat_points,
        regress_ranges=concat_regress_ranges,
        num_points_per_lvl=num_points)

    # split to per img, per level
    labels_list = [labels.split(num_points, 0) for labels in labels_list]
    bbox_targets_list = [
        bbox_targets.split(num_points, 0)
        for bbox_targets in bbox_targets_list
    ]
    angle_targets_list = [
        angle_targets.split(num_points, 0)
        for angle_targets in angle_targets_list
    ]
    num_gts = [len(gt) for gt in gt_bboxes_list]
    gt_idx_list = []
    gt_idx_offset = 0
    for bid, gt_idx in enumerate(_gt_idx_list):
        gt_idx_list.append(
            (gt_idx + gt_idx_offset).split(num_points, 0))
        gt_idx_offset += num_gts[bid]

    # concat per level image
    concat_lvl_labels = []
    concat_lvl_bbox_targets = []
    concat_lvl_angle_targets = []
    concat_lvl_gt_idx = []
    for i in range(num_levels):
        concat_lvl_labels.append(
            torch.cat([labels[i] for labels in labels_list]))
        bbox_targets = torch.cat(
            [bbox_targets[i] for bbox_targets in bbox_targets_list])
        angle_targets = torch.cat(
            [angle_targets[i] for angle_targets in angle_targets_list])
        gt_idx = torch.cat([gt_idx[i] for gt_idx in gt_idx_list])
        if norm_on_bbox:
            bbox_targets = bbox_targets / strides[i]
        concat_lvl_bbox_targets.append(bbox_targets)
        concat_lvl_angle_targets.append(angle_targets)
        concat_lvl_gt_idx.append(gt_idx)
    return (concat_lvl_labels, concat_lvl_bbox_targets,
            concat_lvl_angle_targets, concat_lvl_gt_idx)

def _get_target_single(gt_bboxes, gt_labels, points, regress_ranges,
                        num_points_per_lvl):
    """Compute regression, classification and angle targets for a single
    image."""
    num_points = points.size(0)
    num_gts = gt_labels.size(0)
    if num_gts == 0:
        return gt_labels.new_full((num_points,), num_classes), \
                gt_bboxes.new_zeros((num_points, 4)), \
                gt_bboxes.new_zeros((num_points, 1)), \
                gt_labels.new_full((num_points,), -1),

    areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    areas = areas[None].repeat(num_points, 1)
    regress_ranges = regress_ranges[:, None, :].expand(
        num_points, num_gts, 2)
    points = points[:, None, :].expand(num_points, num_gts, 2)
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
    gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

    cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
    rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                            dim=-1).reshape(num_points, num_gts, 2, 2)
    offset = points - gt_ctr
    offset = torch.matmul(rot_matrix, offset[..., None])
    offset = offset.squeeze(-1)

    w, h = gt_wh[..., 0], gt_wh[..., 1]
    offset_x, offset_y = offset[..., 0], offset[..., 1]
    left = w / 2 + offset_x
    right = w / 2 - offset_x
    top = h / 2 + offset_y
    bottom = h / 2 - offset_y
    bbox_targets = torch.stack((left, top, right, bottom), -1)

    # condition1: inside a gt bbox
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    if center_sampling:
        # condition1: inside a `center bbox`
        radius = center_sample_radius
        stride = offset.new_zeros(offset.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = strides[lvl_idx] * radius
            lvl_begin = lvl_end

        inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                inside_gt_bbox_mask)

    # condition2: limit the regression range for each location
    max_regress_distance = bbox_targets.max(-1)[0]
    inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    areas[inside_gt_bbox_mask == 0] = INF
    areas[inside_regress_range == 0] = INF
    min_area, min_area_inds = areas.min(dim=1)

    labels = gt_labels[min_area_inds]
    labels[min_area == INF] = num_classes  # set as BG
    bbox_targets = bbox_targets[range(num_points), min_area_inds]
    angle_targets = gt_angle[range(num_points), min_area_inds]

    return labels, bbox_targets, angle_targets, min_area_inds


all_level_points = [
    torch.randn(16384, 2),  # shape: [16384, 2]
    torch.randn(4096, 2),   # shape: [4096, 2]
    torch.randn(1024, 2),   # shape: [1024, 2]
    torch.randn(256, 2),    # shape: [256, 2]
    torch.randn(64, 2)      # shape: [64, 2]
]

gt_bboxes = [
    torch.randn(44, 5),  # 第一张图像有 44 个边界框，每个边界框 5 个值
    torch.randn(1, 5)  # 第二张图像有 1 个边界框，每个边界框 5 个值
]

gt_labels = [
    torch.randn(44),  # 第一张图像有 44 个边界框，每个边界框 5 个值
    torch.randn(1)   # 第二张图像有 1 个边界框，每个边界框 5 个值
]

labels, bbox_targets, angle_targets, gt_idx = get_targets(
    all_level_points, gt_bboxes, gt_labels)


bbox_targets = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5],
                             [5.0, 6.0, 7.0, 8.0, 1.0],
                             [9.0, 10.0, 11.0, 12.0, 1.5]])
min_area_inds = torch.tensor([0, 2, 1])
bbox_targets = bbox_targets[range(3), min_area_inds]
print(bbox_targets)