array=(
	"trial1"
	"trial2"
	"trial3"
	"trial4"
	"trial5"
	"trial6"
	"trial7"
	"trial8"
	"trial9"
	"trial10"
)

# for idx in $( seq 1 10 )
# do
# 	CUDA_VISIBLE_DEVICES=0 python -W ignore -u train_base.py \
# 					--data_root "/path/to/your/RegDB/" \
# 					--dataset "regdb" \
# 					--save "./checkpoints/" \
# 					--print_freq 5 \
# 					--test_freq 10 \
# 					--epochs 120 \
# 					--batch_size 64 \
# 					--test_batch 128 \
# 					--num_pos 4 \
# 					--optim 'adam' \
# 					--lr 0.01 \
# 					--beta1 0.5 \
# 					--beta2 0.999 \
# 					--weight_decay 5e-4 \
# 					--img_w 128 \
# 					--img_h 256 \
# 					--label_smooth 0.0 \
# 					--last_stride 1 \
# 					--dropout_rate 0.0 \
# 					--ema_decay 0.0 \
# 					--sp_lambda 0.0 \
# 					--cmmd_lambda 0.0 \
# 					--margin 0.4 \
# 					--triplet_feat_norm 'no' \
# 					--test_feat_norm 'yes' \
# 					--trial $idx \
# 					--note 'regdb-baseline/'${array[$idx]}
# done


for idx in $( seq 1 10 )
do
	CUDA_VISIBLE_DEVICES=0 python -W ignore -u train_eval.py \
					--data_root "/path/to/your/RegDB/" \
					--dataset "regdb" \
					--save "./checkpoints/" \
					--config_path "./checkpoints_regdb/cm-nas_searched_arch/config.cfg" \
					--print_freq 5 \
					--test_freq 10 \
					--epochs 120 \
					--batch_size 64 \
					--test_batch 128 \
					--num_pos 4 \
					--optim 'adam' \
					--lr 0.01 \
					--beta1 0.5 \
					--beta2 0.999 \
					--weight_decay 5e-4 \
					--img_w 128 \
					--img_h 256 \
					--label_smooth 0.0 \
					--last_stride 1 \
					--dropout_rate 0.0 \
					--ema_decay 0.968 \
					--sp_lambda 5.0 \
					--cmmd_lambda 0.05 \
					--margin 0.4 \
					--triplet_feat_norm 'no' \
					--test_feat_norm 'yes' \
					--trial $idx \
					--note 'regdb-eval-ema/'${array[$idx]}
done
