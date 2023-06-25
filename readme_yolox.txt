
pip install sklearn pandas -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 复现COCO上的精度 -----------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolox/yolox_s_300e_coco.yml --amp --eval     > yolox_s_300e_coco.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/yolox/yolox_s_300e_coco.yml --amp --eval     > yolox_s_300e_coco.log 2>&1 &



python tools/train.py -c configs/yolox/yolox_s_300e_coco.yml --amp --eval





