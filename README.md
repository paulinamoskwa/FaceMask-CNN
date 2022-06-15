# **Face Mask CNN-Classifier with `TensorFlow` e `Keras`**

<p align="center">
<img src='./Miscellaneous/img00.png' width='700'/>
</p>

## 📖 **About**

The aim of this project is to build an automatic classifier that can recognize whether in a photo all the people are wearing masks, only some of them, or none at all. The dataset is available on [ kaggle](https://www.kaggle.com/competitions/artificial-neural-networks-and-deep-learning-2020). It is an artificial dataset (of about 5.614 images), so some images are particularly problematic (or, otherwise, funny), for example:

<p align="center">
<img src='./Miscellaneous/img01.png' width='400'/><br>
<i>In the first line there are problematic images.</i>
</p>

## 📝 **Approach and results**
Through the use of `TensorFlow` and `Keras`, I build an ensemble of convolutional neural networks for classification. 

First of all, `TensorFlow` offers three different options for reading the dataset as input, therefore, I initially decided to test the efficiency of the three options:
* `flow_from_directory`
* `flow_from_directory` and `Dataset.from_generator`
* `image_dataset_from_directory`

I created a convolutional neural network for each of the three options and compared the result. Obviously the three options affect the whole code and (as you can see below) the training results as well. While changing the way of importing the images, I have kept the structure of the neural network the same, so that I could make a fair comparison. In the notebooks [Face Mask 1](https://github.com/PaulinoMoskwa/FaceMask-CNN/blob/master/Notebooks/Face%20Mask%201%20-%20flow_from_directory.ipynb), [Face Mask 2](https://github.com/PaulinoMoskwa/FaceMask-CNN/blob/master/Notebooks/Face%20Mask%202%20-%20flow_from_directory%2C%20Dataset.from_generator.ipynb) and [Face Mask 3](https://github.com/PaulinoMoskwa/FaceMask-CNN/blob/master/Notebooks/Face%20Mask%203%20-%20image_dataset_from_directory.ipynb) are shown all the details. 

The three notebooks are structured in the same way:
* Import of the dataset 
* Dataset exploration and classes analysis
* Base model definition (via [`Keras`](https://keras.io/api/applications/))
* Compile and train of the model
* Model performance evaluation
* Prediction
* GradCAM

### 🔎 **Import of the dataset**
The method by which the dataset is imported also affects the definition of the loss function, since it changes the encoding of the labels. In the case of `flow_from_directory` the labels are one-hot-encoded (e.g. for a 3-class classification the labels are: `[1,0,0]` , `[0,1,0]`, `[0,0,1]`) and this requires the loss function to be `CategoricalCrossentropy()`. By contrast, in the case of `image_dataset_from_directory`, the labels are encoded as integers (e.g. for a 3-class classifications the labels are: `[1]` , `[2]`, `[3]`) and this requires the loss function to be `SparseCategoricalCrossentropy()`.

### 🔎 **Classes analysis**
An important factor to consider when it comes to classification is the balance of the classes. In this case the three classes (*all masks*, *no masks*, *some masks*) are adequately balanced among each other. 

### 🔎 **Model performance evaluation**
To figure out which of the three approaches is the most efficient, I relied on several elements.

#### **1. Accuracy**

<table align="center">
    <tr>
        <th><div align="left">   Import Option </div></th>
        <th><div align="center"> Train Accuracy </div></th>
        <th><div align="center"> Validation Accuracy </div></th>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code></div></td>
        <td><div align="center"> 0.9913 </div></td>
        <td><div align="center"> 0.8575 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code>, <code> Dataset.from_generator</code></div></td>
        <td><div align="center"> 0.9906 </div></td>
        <td><div align="center"> 0.8498 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> image_dataset_from_directory </code></div></td>
        <td><div align="center"> 0.9995 </div></td>
        <td><div align="center"> 0.9133 </div></td>
    </tr>
</table>

The third option proved to be the most efficient not only because of the highest validation accuracy value, but also because of the smallest gap between train and validation accuracy. 

#### **2. Confusion matrix**
From left to right: `flow_from_directory`, `flow_from_directory` and `Dataset.from_generator`, `image_dataset_from_directory`:

<p align="center">
<img src='./Miscellaneous/img02.png' width='800'/>
</p>

Also with this evaluation criterion, the third option turned out to be the best.

#### **3. Quantitative indeces**
*Version*: Micro<br>
It calculates metrics globally by counting the total true positives, false negatives and false positives.

<table align="center">
    <tr>
        <th><div style="text-align:left;">   Import Option </div></th>
        <th><div style="text-align:center;"> Precision </div></th>
        <th><div style="text-align:center;"> Recall </div></th>
        <th><div style="text-align:center;"> Accuracy </div></th>
        <th><div style="text-align:center;"> F1 </div></th>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code></div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code>, <code> Dataset.from_generator</code></div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> image_dataset_from_directory </code></div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
    </tr>
</table>

*Version*: Macro<br>
It calculates metrics for each label, and finds their unweighted mean. This does not take label imbalance into account.

<table align="center">
    <tr>
        <th><div style="text-align:left;">   Import Option </div></th>
        <th><div style="text-align:center;"> Precision </div></th>
        <th><div style="text-align:center;"> Recall </div></th>
        <th><div style="text-align:center;"> Accuracy </div></th>
        <th><div style="text-align:center;"> F1 </div></th>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code></div></td>
        <td><div style="text-align:center;"> 0.85551 </div></td>
        <td><div style="text-align:center;"> 0.85565 </div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
        <td><div style="text-align:center;"> 0.85454 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code>, <code> Dataset.from_generator</code></div></td>
        <td><div style="text-align:center;"> 0.84730 </div></td>
        <td><div style="text-align:center;"> 0.84810 </div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
        <td><div style="text-align:center;"> 0.84723 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> image_dataset_from_directory </code></div></td>
        <td><div style="text-align:center;"> 0.97375 </div></td>
        <td><div style="text-align:center;"> 0.97348 </div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
        <td><div style="text-align:center;"> 0.97356 </div></td>
    </tr>
</table>

*Version*: Weighted<br>
It calculates metrics for each label, and finds their average weighted by the number of true instances for each label. This alters 'macro' to account for label imbalance.

<table align="center">
    <tr>
        <th><div style="text-align:left;">   Import Option </div></th>
        <th><div style="text-align:center;"> Precision </div></th>
        <th><div style="text-align:center;"> Recall </div></th>
        <th><div style="text-align:center;"> Accuracy </div></th>
        <th><div style="text-align:center;"> F1 </div></th>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code></div></td>
        <td><div style="text-align:center;"> 0.85617 </div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
        <td><div style="text-align:center;"> 0.85748 </div></td>
        <td><div style="text-align:center;"> 0.85580 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> flow_from_directory </code>, <code> Dataset.from_generator</code></div></td>
        <td><div style="text-align:center;"> 0.84818 </div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
        <td><div style="text-align:center;"> 0.84976 </div></td>
        <td><div style="text-align:center;"> 0.84851 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;"><code> image_dataset_from_directory </code></div></td>
        <td><div style="text-align:center;"> 0.97385 </div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
        <td><div style="text-align:center;"> 0.97387 </div></td>
        <td><div style="text-align:center;"> 0.97381 </div></td>
    </tr>
</table>

In all cases and in all indices, the third option prevails.

### 🔎 **GradCAM**
This section was not actually introduced for the purpose of adding anything to the comparison between the three methods of importing images. Instead, it was included as a matter of 'Explainable AI'. As good as a model can be as a classifier, it is necessary that its decisions be justified in some way. Through the introduction of GradCAM, it is possible to understand which section of the photo the model focused on to make its prediction. 

<p align="center">
<img src='./Miscellaneous/img03.png' width='500'/><br>
<i>The more red an area is, the more it was considered during the final prediction.</i>
</p>

<br>

<p align="center">
<img src='./Miscellaneous/img04.png' width='700'/><br>
<i>Predictions with the most important areas for each image.</i>
</p>

---------------------

Once the first part of the work was completed, I moved onto creating the ensemble. Using the third option as the method for importing the images (`image_dataset_from_directory`), I trained seven models independently of each other. The results are reported in the notebook [Ensemble 1 - Training](https://github.com/PaulinoMoskwa/FaceMask-CNN/blob/master/Notebooks/Ensemble%201%20-%20Training.ipynb) and also in the table:

<table align="center">
    <tr>
        <th><div style="text-align:left;">   Base Model </div></th>
        <th><div style="text-align:center;"> Train Accuracy </div></th>
        <th><div style="text-align:center;"> Validation Accuracy </div></th>
    </tr>
    <tr>
        <td><div style="text-align:left;">   EfficientNetB3 </div></td>
        <td><div style="text-align:center;"> 0.9916 </div></td>
        <td style="background-color:green;color:white;"><div style="text-align:center;"> 0.9305 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">   Xception </div></td>
        <td style="background-color:red;color:white;"><div style="text-align:center;"> 0.8858 </div></td>
        <td style="background-color:red;color:white;"><div style="text-align:center;"> 0.7844 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">   ResNet50V2 </div></td>
        <td><div style="text-align:center;"> 0.9529 </div></td>
        <td><div style="text-align:center;"> 0.8325 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">   InceptionV3 </div></td>
        <td style="background-color:green;color:white;"><div style="text-align:center;"> 0.9977 </div></td>
        <td><div style="text-align:center;"> 0.8931 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">   InceptionResNetV2 </div></td>
        <td><div style="text-align:center;"> 0.9944 </div></td>
        <td><div style="text-align:center;"> 0.8996 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">   DenseNet201 </div></td>
        <td><div style="text-align:center;"> 0.9399 </div></td>
        <td><div style="text-align:center;"> 0.8901 </div></td>
    </tr>
    <tr>
        <td><div style="text-align:left;">   EfficientNetV2S </div></td>
        <td><div style="text-align:center;"> 0.9702 </div></td>
        <td><div style="text-align:center;"> 0.8967 </div></td>
    </tr>
</table>

From these seven neural networks I created a model that, taking an image as input, returns as a class the one that was predicted the most times among the seven networks. The results are reported in the notebook [Ensemble 2 - Prediction](https://github.com/PaulinoMoskwa/FaceMask-CNN/blob/master/Notebooks/Ensemble%202%20-%20Prediction.ipynb).
