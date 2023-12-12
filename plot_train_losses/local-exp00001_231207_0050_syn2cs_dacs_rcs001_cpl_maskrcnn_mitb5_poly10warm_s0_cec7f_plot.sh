#!/bin/bash

# exit when any command fails
set -e

JSON_FILE_PATH=losses/losses_experiments_7_edaps/exp-00001/work_dirs/local-exp00001/231207_0050_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_cec7f/20231207_005017.log.json
OUT_FILE_LOSS_DISC=losses/losses_experiments_7_edaps/exp-00001/work_dirs/local-exp00001/231207_0050_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_cec7f/loss_disc.pdf
OUT_FILE_LOSS_GEN=losses/losses_experiments_7_edaps/exp-00001/work_dirs/local-exp00001/231207_0050_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_cec7f/loss_gen.pdf
OUT_FILE_LOSS_PIX=losses/losses_experiments_7_edaps/exp-00001/work_dirs/local-exp00001/231207_0050_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_cec7f/loss_pix.pdf
OUT_FILE_LOSS_TOTAL=losses/losses_experiments_7_edaps/exp-00001/work_dirs/local-exp00001/231207_0050_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_cec7f/loss_total.pdf

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
