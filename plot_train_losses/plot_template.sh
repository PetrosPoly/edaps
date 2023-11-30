#!/bin/bash

# exit when any command fails
set -e

JSON_FILE_PATH={json_file_path}
OUT_FILE_LOSS_DISC={out_file_loss_disc}
OUT_FILE_LOSS_GEN={out_file_loss_gen}
OUT_FILE_LOSS_PIX={out_file_loss_pix}
OUT_FILE_LOSS_TOTAL={out_file_loss_total}

python tools/analysis_tools_mmdet/analyze_logs.py plot_curve \
      $JSON_FILE_PATH \
    --keys decode.loss_seg rpn.loss_rpn_cls rpn.loss_rpn_bbox roi.loss_cls roi.loss_bbox roi.loss_mask src.loss_imnet_feat_dist mix.decode.loss_seg\
    --legend decode.loss_seg rpn.loss_rpn_cls rpn.loss_rpn_bbox roi.loss_cls roi.loss_bbox roi.loss_mask src.loss_imnet_feat_dist mix.decode.loss_seg \
    --out $OUT_FILE_LOSS_DISC
    
python tools/analysis_tools_mmdet/analyze_logs.py plot_curve \
      $JSON_FILE_PATH \
    --keys decode.loss_seg contrastive_loss rpn.loss_rpn_cls rpn.loss_rpn_bbox roi.loss_cls roi.loss_bbox roi.loss_mask src.loss_imnet_feat_dist mix.decode.loss_seg\
    --legend decode.loss_seg contrastive_loss rpn.loss_rpn_cls rpn.loss_rpn_bbox roi.loss_cls roi.loss_bbox roi.loss_mask src.loss_imnet_feat_dist mix.decode.loss_seg\
    --out $OUT_FILE_LOSS_DISC

python tools/analysis_tools_mmdet/analyze_logs.py plot_curve \
      $JSON_FILE_PATH \
    --keys loss_gan_g \
    --legend loss_gan_g \
    --out $OUT_FILE_LOSS_GEN

python tools/analysis_tools_mmdet/analyze_logs.py plot_curve \
      $JSON_FILE_PATH \
    --keys pixel_loss \
    --legend pixel_loss \
    --out $OUT_FILE_LOSS_PIX

python tools/analysis_tools_mmdet/analyze_logs.py plot_curve \
      $JSON_FILE_PATH \
    --keys loss \
    --legend loss \
    --out $OUT_FILE_LOSS_TOTAL


#python tools/analysis_tools/analyze_logs.py plot_curve \
#      /media/suman/CVLHDD/apps/experiments/mmgeneration_experiments_on_euler_backup/euler-exp00002/231015_2328_lamp_pix2pix_02d15/20231015_232956.log.json \
#    --keys loss \
#    --legend loss \
#    --out /media/suman/CVLHDD/apps/experiments/mmgeneration_experiments_on_euler_backup/euler-exp00002/231015_2328_lamp_pix2pix_02d15/loss_total.pdf
