CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/analysis_tools/benchmark.py configs/csl/rotated_retinanet_obb_csl_gaussian_r50_adamw_fpn_fp16_1x_dota_le90.py work_dirs/rotated_retinanet_obb_csl_gaussian_r50_adamw_fpn_fp16_1x_dota_le90/latest.pth --launcher pytorch