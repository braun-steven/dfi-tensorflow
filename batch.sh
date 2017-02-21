#!/bin/bash
cd dfi-tensorflow
git pull
cd ..
ALPHA_ARR=(0.4 1.0)
K_ARR=(100 200)
PERSON_ARR=("Will_Smith/Will_Smith_0001.jpg" "Leonardo_DiCaprio/Leonardo_DiCaprio_0001.jpg" "Donald_Trump/Donald_Trump_0001.jpg" "Vladimir_Putin/Vladimir_Putin_0002.jpg" "Bill_Clinton/Bill_Clinton_0005.jpg" "George_W_Bush/George_W_Bush_0016.jpg" "Silvio_Berlusconi/Silvio_Berlusconi_0023.jpg" "Gerhard_Schroeder/Gerhard_Schroeder_0065.jpg")
FEATS=("Asian" "Black" "Baby" "Child" "Youth" "Middle_Aged" "Senior" "Bald" "No_Eyewear" "Eyeglasses" "Sunglasses" "Mustache" "Smiling" "Frowning" "Chubby" "Bangs" "Sideburns" "Bushy_Eyebrows" "Arched_Eyebrows" "Narrow_Eyes" "Eyes_Open" "Big_Nose" "Pointy_Nose" "Big_Lips" "Mouth_Closed" "Mouth_Slightly_Open" "Mouth_Wide_Open" "No_Beard" "Goatee" "Round_Jaw" "Double_Chin" "Wearing_Hat" "Square_Face" "Round_Face" "Attractive_Man" "Attractive_Woman" "Indian" "Bags_Under_Eyes" "Heavy_Makeup" "Rosy_Cheeks" "Strong_Nose-Mouth_Lines" "Wearing_Lipstick" "Flushed_Face" "High_Cheekbones" "Wearing_Earrings" "Wearing_Necktie" "Wearing_Necklace")
for ALPHA in "${ALPHA_ARR[@]}"
do
	for P in "${PERSON_ARR[@]}"
	do
	    for FEAT in "${FEATS[@]}"
	    do 
		for K in "${K_ARR[@]}"
		do
	    		./env/bin/python dfi-tensorflow/src/main.py -d dfi-tensorflow/data/ \
				-m vgg19.npy \
				-g \
				--optimizer adam \
				--steps 2000 \
				--lr 0.1 \
				--rebuild-cache \
				--k $K \
				--alpha $ALPHA \
				--beta 2 \
				--lamb 0.001 \
				--person-image dfi-tensorflow/data/lfw-deepfunneled/$P \
				--output out \
				-f $FEAT
		done
	    done
	done
done
