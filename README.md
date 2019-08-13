# A tensorflow implementation of fire-FRD-CNN network.

### To training the model, the  prerequisites are:
1. python 2.7
2. easydict
3. joblib
4. numpy
5. opencv-python
6. Pillow
7. tensorflow-gpu

### Please follow the belowing steps to training the model.

- Download KITTI object detection dataset. Put them under `$fire-FRD-CNN_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$fire-FRD-CNN_ROOT/data/KITTI/training/` and `$fire-FRD-CNN_ROOT/data/KITTI/testing/`. 

- Now we need to split the training data into a training set and a vlidation set. 

  ```Shell
  cd $fire-FRD-CNN_ROOT/data/KITTI/
  mkdir ImageSets
  cd ./ImageSets
  ls ../training/image_2/ | grep ".png" | sed s/.png// > trainval.txt
  ```
  `trainval.txt` contains indices to all the images in the training data. In our experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. For your convenience, we provide a script to split the train-val set automatically. Simply run
  
  ```Shell
  cd $fire-FRD-CNN_ROOT/data/
  python random_split_train_val.py
  ```
  
  then you should get the `train.txt` and `val.txt` under `$fire-FRD-CNN_ROOT/data/KITTI/ImageSets`. 

  When above two steps are finished, the structure of `$fire-FRD-CNN_ROOT/data/KITTI/` should at least contain:

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```

- Next, download the CNN model pretrained for ImageNet classification:
  ```Shell
  cd $fire-FRD-CNN_ROOT/data/
  # SqueezeNet
  wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
  tar -xzvf SqueezeNet.tgz

- Now we can start training. Training script can be found in `$fire-FRD-CNN_ROOT/scripts/train.sh`, which contains commands to train the model: fire-FRD-CNN
  ```Shell
  cd $fire-FRD-CNN_ROOT/
  ./scripts/train.sh -net fire-FRD-CNN -train_dir /home/scott/logs/fire-FRD-CNN -gpu 0
  ```

  Training logs are saved to the directory specified by `-train_dir`. GPU id is specified by `-gpu`. Network to train is specificed by `-net` 

- Before evaluation, you need to first compile the official evaluation script of KITTI dataset
  ```Shell
  cd $fire-FRD-CNN/src/dataset/kitti-eval
  make
  ```

- Then, you can launch the evaluation script (in parallel with training) by 

  ```Shell
  cd $fire-FRD-CNN/
  ./scripts/eval.sh -net fire-FRD-CNN -eval_dir /home/scott/logs/fire-FRD-CNN -image_set (train|val) -gpu 1
  ```
