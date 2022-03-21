Code for the paper "Galaxy Morphology Classification using Neural Ordinary Differential Equations" (https://arxiv.org/pdf/2012.07735.pdf)

Here we compare Resnet, NODE and NODE_ACA for galaxy-morphology classfication problem.

This repo is forked from https://github.com/juntang-zhuang/torch_ACA

For dataset, we use subset of https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge. We pick galaxy classifications with high probability only, as described in this paper https://arxiv.org/pdf/1308.3496.pdf. Thus, the original kaggle dataset of 60,000 images is reduced to 28,790 images, with 5 classes viz spiral, edge-on, cigar-shaped smooth, in-between smooth, and completely round smooth.

Final pruned dataset with 28,790 classes can be found at following link https://drive.google.com/drive/folders/1xsgMao2ozDQ0BJfH5QoEMgvdtxuqCaJu?usp=sharing.

Got the highest accuracy of ~93% with following parameters -: 
python experiments/train.py  --data galaxyzoo --optimizer adam --lr 0.001  --model resnet  --batch_size 32 --test_batch_size 32 --num_epochs 100







