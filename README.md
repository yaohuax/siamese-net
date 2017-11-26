# siamese-net
This network is uesed to compute the dissimilarity between two faces.

p1a.py uses BCELoss function while p1b.py uses contrastive loss function.

To run the network and save the model

python p1a.py --save MODEL_NAME

To test the model

python p1a.py --load MODEL_NAME
