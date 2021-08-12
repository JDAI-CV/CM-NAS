ROOT="./checkpoints_regdb/eval-cm-nas_searched_arch--sp-lam5.0-cmmd-lam0.05-ema/"
CONFIG="./checkpoints_regdb/cm-nas_searched_arch/config.cfg"

array=(
	"trial1/checkpoint120.pth.tar"
	"trial2/checkpoint120.pth.tar"
	"trial3/checkpoint120.pth.tar"
	"trial4/checkpoint120.pth.tar"
	"trial5/checkpoint120.pth.tar"
	"trial6/checkpoint120.pth.tar"
	"trial7/checkpoint120.pth.tar"
	"trial8/checkpoint120.pth.tar"
	"trial9/checkpoint120.pth.tar"
	"trial10/checkpoint120.pth.tar"
)

for idx in $( seq 1 10 )
do
	WEIGHTS=${array[$idx]}; 
	echo $WEIGHTS;
	CUDA_VISIBLE_DEVICES=0 python -W ignore -u test.py \
						--data_root "/path/to/your/RegDB/" \
						--dataset "regdb" \
						--model_type "cm-nas" \
						--config $CONFIG \
						--weights $ROOT$WEIGHTS \
						--test_batch 128 \
						--img_w 128 \
						--img_h 256 \
						--last_stride 1 \
						--test_feat_norm 'yes' \
						--mode 'all' \
						--shot 1 \
						--trial $idx \
						# --tvsearch \
						# --ema
done

