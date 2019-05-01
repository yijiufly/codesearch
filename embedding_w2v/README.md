# NN_embedding
Neural Network model for generation graph embedding and use it for vulnerability detection using siamese network


# Important Files Description
siamese_emb.py : Contains Siamese class.  'emb_generation' function takes acfg as input and generates its embedding.
'siamese_cosine_loss' function takes two embedding as inputs and gives l2_loss as output.
'loss_with_spring' & 'loss_with_step' are different version of eucledian distance calculation but is not used.

emb_train.py : file where model is trained i.e tensorflow session is created and executed.

dataset.py : file contains methods to generate input for tesorflow model

preprocess.py : this file contains the code to extract raw features of the desired binary file and save the in '.ida' foramt in 'out_analysis' folder. These '.ida' files will be used to generate the input to train and test the NN.

util.py : this file contains utility functions to be used in mutliple files

raw_feature_extractor folder contains files which helps in extracting features of binary and mainly used by preprocess.py file

raw_graphs.py : this files contains most of the information about the raw_graph which is being extracted

preprocessing_ida.py : this file is the script which is executed by ida pro to generate raw_graph_list and saves in '.ida' file
