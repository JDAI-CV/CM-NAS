WEIGHTS="./checkpoints_sysu/eval-cm-nas_searched_arch-sp-lam5.0-cmmd-lam0.05-ema/checkpoint.pth.tar"
CONFIG="./checkpoints_sysu/cm-nas_searched_arch/config.cfg"

CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
						--data_root "/path/to/your/SYSU-MM01/" \
						--dataset "sysu" \
						--model_type "cm-nas" \
						--config_path $CONFIG \
						--weights $WEIGHTS \
						--test_batch 128 \
						--img_w 128 \
						--img_h 256 \
						--last_stride 1 \
						--test_feat_norm 'yes' \
						--mode 'all' \
						--shot 1 \
						# --ema


CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
						--data_root "/path/to/your/SYSU-MM01/" \
						--dataset "sysu" \
						--model_type "cm-nas" \
						--config_path $CONFIG \
						--weights $WEIGHTS \
						--test_batch 128 \
						--img_w 128 \
						--img_h 256 \
						--last_stride 1 \
						--test_feat_norm 'yes' \
						--mode 'all' \
						--shot 10 \
						# --ema


CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
						--data_root "/path/to/your/SYSU-MM01/" \
						--dataset "sysu" \
						--model_type "cm-nas" \
						--config_path $CONFIG \
						--weights $WEIGHTS \
						--test_batch 128 \
						--img_w 128 \
						--img_h 256 \
						--last_stride 1 \
						--test_feat_norm 'yes' \
						--mode 'indoor' \
						--shot 1 \
						# --ema


CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
						--data_root "/path/to/your/SYSU-MM01/" \
						--dataset "sysu" \
						--model_type "cm-nas" \
						--config_path $CONFIG \
						--weights $WEIGHTS \
						--test_batch 128 \
						--img_w 128 \
						--img_h 256 \
						--last_stride 1 \
						--test_feat_norm 'yes' \
						--mode 'indoor' \
						--shot 10 \
						# --ema

