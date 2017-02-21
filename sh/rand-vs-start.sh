#!/bin/bash
cd dfi-tensorflow
git pull
cd ..
ALPHA_ARR=(0.4)
PERSON_ARR=("Silvio_Berlusconi/Silvio_Berlusconi_0023.jpg")
FEATS=("Senior")
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
				--steps 6000 \
				--lr 0.1 \
				--rebuild-cache \
				--k 200 \
				--alpha $ALPHA \
				--beta 2 \
				--lamb 0.001 \
				--person-image dfi-tensorflow/data/lfw-deepfunneled/$P \
				--output rand-vs-start \
				--discrete-knn \
				-f $FEAT
	    done
	done
done
