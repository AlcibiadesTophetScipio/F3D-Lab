export CUDA_VISIBLE_DEVICES=0

# trian
python ../runComa.py --run_type train

# train from lastest
python ../runComa.py --run_type train \
                      --load_epoch -1

# train from specified epoch
python ../runComa.py --run_type train \
                      --load_epoch 20

# eval from latest
python ../runComa.py --run_type eval \
                      --load_epoch -1

# eval from specified epoch
python ../runComa.py --run_type eval \
                      --load_epoch 20

# reconstruct from latest
python ../runComa.py --run_type rec \
                      --load_epoch -1

# reconstruct from specified epoch
python ../runComa.py --run_type rec \
                      --load_epoch 20