# MaskDetection
# UseCase 
In these trying times, we need to ensure not only that we are wearing mask but also others around us have taken this precaution. We have security cameras placed at various public places. This project was to help identify people who are not wearing mask properly or not wearing them at all. By checking people in real time, appropriate action can be taken immediately.

# DataSet 
Used kaggle dataset https://www.kaggle.com/andrewmvd/face-mask-detection for  the same. 
Though the dataset is small(Wanted to annotate some images myself using labelstud.io but couldn't)
Dataset is well annotated with 3 classes. 
    - No Mask 
    - Improperly wearing Mask 
    - Wearing Mask

# Products Used 
   * **Pytorch Lightening:** Excellent framework for structuring your project. Tried various features              like checkpoints, gpu support etc.
   * **Pytorch:** Python ibrary used to build the model
   * **Google Colab:** Used to train the model as the GPU is not available locally.
   * **Weight & Biases:** Used to monitor the training progress.

# Model Used: 
Tried with MASK-RCNN using a pretrained model but could use it properly with PyTorch lightening. So tried to train a __YOLO v3 model__ on custom dataset. 

# Results:
Couldn't get the model to converge. Though completed the full model development and prediction handling for new data.

<img width="923" alt="Result" src="https://user-images.githubusercontent.com/17312814/118266209-e5204e00-b4d7-11eb-94d7-e5e47ee05f3b.png">
