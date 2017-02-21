#!/bin/bash
cd dfi-tensorflow
git pull
cd ..
ALPHA_ARR=(0.45)
PERSON_ARR=("Silvio_Berlusconi/Silvio_Berlusconi_0023.jpg")
FEATS=("Sunglasses")
for ALPHA in "${ALPHA_ARR[@]}"
do
	for P in "${PERSON_ARR[@]}"
	do
	    for FEAT in "${FEATS[@]}"
	    do 
	    		./env/bin/python dfi-tensorflow/src/main.py -d dfi-tensorflow/data/ \
				-m vgg19.npy \
				-g \
				--optimizer adam \
				--steps 2000 \
				--lr 0.1 \
				--rebuild-cache \
				--k 100 \
				--alpha $ALPHA \
				--beta 2 \
				--lamb 0.001 \
				--person-image dfi-tensorflow/data/lfw-deepfunneled/$P \
				--output berlusc-sunglasses \
				--discrete-knn \
				-f $FEAT
	    done
	done
done
