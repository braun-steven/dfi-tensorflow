./env/bin/python dfi-tensorflow/src/main.py -d dfi-tensorflow/data/ \
                -m vgg19.npy \
                -g \
                --optimizer adam \
                --steps 1000 \
                --lr 0.1 \
                --rebuild-cache \
                --k 200 \
		--lamb 0.001 \
                --beta 2 \
                --alpha 0.4 \
                -f 'Mustache' \
		--person-image dfi-tensorflow/data/lfw-deepfunneled/Donald_Trump/Donald_Trump_0001.jpg \
	
