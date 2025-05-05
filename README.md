# AI-Image-Generator
Dataset Link - https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
Instructions-
1. In order to run notebook one need to extract the dataset outside this project folder(AI_Image_Generator).
2. User Interface for this project to be created in the 'User Interface' folder.
For testing:
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python -m spacy download en_core_web_sm
command:
pip install numpy pandas tqdm tensorflow==2.18
Early fusion model -
image and text embedding are concatenated befor feeding to lstm
repeated vector of image to make compatible dimension for concatenating text embedding
mask_zero=true is not feasible in this approach as masked text is concatenated with unmasked image repeated vector
Epoch 20/50
2110/2110 ━━━━━━━━━━━━━━━━━━━━ 0s 201ms/step - loss: 3.5287
Epoch 20: val_loss did not improve from 4.06468
2110/2110 ━━━━━━━━━━━━━━━━━━━━ 506s 239ms/step - loss: 3.5287 - val_loss: 4.0796 - learning_rate: 4.0000e-05

Later fusion model -
image and text embedding are added after processing text embedding in lstm
