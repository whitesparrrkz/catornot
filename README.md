# catornot

This is a very simple image classifier model made using **PyTorch**. This was made to classify if an image was a cat or not, but it is easily extenible by adding more folders & training data.  

## train.py

When running train.py, it will use the directory **img_dataset** as the training dataset. There should be two sub directories in that folder, called test and train.  

The subfolders within these two directories are then the types of classifications that the model will trian for **(EX: cats folder, and notcats folder)**.  
Then, just load the folders with jpg images as training data. (I used [cats dataset](https://www.kaggle.com/datasets/crawford/cat-dataset?resource=download) and [notcats (scenery) dataset](https://www.kaggle.com/datasets/pankajkumar2002/random-image-sample-dataset))  

After running, it should create a **catornot_model_weights.pth** file, which is the finished product of the trained model. With this file, predict.py can be run.

## predict.py

To use predict.py, make sure train.py has been ran first, and then run:
```
python predict.py {path_to_img}
```
This command will try and predict what class the image if apart of. 

## examples:  

Heres a cat example:  
  
<img src="https://i.imgur.com/vnKM7oS.jpeg" alt="a cute cat" width="400">
<img src="https://i.imgur.com/4IfDRsH.png" alt="its a cat!">

---

Heres a random (scenery) example:  
  
<img src="https://i.imgur.com/ybT0p6g.jpeg" alt="a cool forest" width="400" height="400">
<img src="https://i.imgur.com/tkRrcoT.png" alt="its not a cat!">

---

However, since I only used a cats image dataset and a scenery dataset, it can produce some **interesting** results.  
  
<img src="https://i.imgur.com/3QsQxmr.jpeg" alt="dr. house (not a cat)" width="400">
<img src="https://i.imgur.com/aIBtBA7.png" alt="well, maybe he is a cat?">

This can easily be fixed though by using a bigger variety of datasets, such as including people, objects, and animals (except cats) to better identify if something is a cat or not.
