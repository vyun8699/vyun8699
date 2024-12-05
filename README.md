# 🔭 Vincent Yunansan, MBA CFA 


I'm a former investment banking and private equity professional with over 7 years of experience who has taken the leap into the world of data science. 
I took the Master's in Data Science program at USYD as a career break to explore different problem sets in analytics and machine learning. 

I am most interested in building repeatable/scalable solutions to process large amounts of data while maintaining simplicity. 

 My latest CV can be accessed here: <a href="assets/Resume_Vincent_Yunansan.pdf"> Curriculum Vitae</a> 


Have any questions? Below is my contact information:
- Email &ensp;&ensp;: vincent.yunansan@gmail.com  
- Phone &ensp;: +61459961345
</br>



## Project Showcase - Table of Contents

| Project | Tools and models |
|---------|--------------|
| [No-code EDA Platform](#no-code-exploratory-data-analysis-platform) | ![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
| [Revenue Optimization with Batching](#revenue-optimization-for-an-australian-commodity-producer) | ![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white) ![GCP](https://img.shields.io/badge/-GCP-yellow?style=flat&logo=Google&logoColor=white) <br>![Static Badge](https://img.shields.io/badge/BFS-grey) ![Static Badge](https://img.shields.io/badge/MILP-grey) |
| [Small Object Detection for a Drone](#small-object-detection-model-for-an-australian-drone-company) | ![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white) ![Pytorch](https://img.shields.io/badge/-Pytorch-orange?style=flat&logo=pytorch&logoColor=white) ![Pytorch](https://img.shields.io/badge/-TensorFlow-red?style=flat&logo=tensorflow&logoColor=white) ![Static Badge](https://img.shields.io/badge/Edge_hardware-white) <br>![Static Badge](https://img.shields.io/badge/YOLO-grey) ![Static Badge](https://img.shields.io/badge/SSD-grey) ![Static Badge](https://img.shields.io/badge/FRCNN-grey) ![Static Badge](https://img.shields.io/badge/SAM-grey) ![Static Badge](https://img.shields.io/badge/FRCNN-grey) ![Static Badge](https://img.shields.io/badge/SAHI-grey)  |
| [Multi-class Object Classification](#multi-class-object-classification-via-transfer-learning) | ![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white) ![Tableau](https://img.shields.io/badge/-Tableau-FF0000?style=flat&logo=tableau&logoColor=white) ![Pytorch](https://img.shields.io/badge/-Pytorch-orange?style=flat&logo=pytorch&logoColor=white) ![Pytorch](https://img.shields.io/badge/-Google_Colab-yellow?style=flat&logo=googlecolab&logoColor=white) <br>![Static Badge](https://img.shields.io/badge/GoogLeNet-grey) ![Static Badge](https://img.shields.io/badge/ResNext-grey) ![Static Badge](https://img.shields.io/badge/Shufflenet-grey) ![Static Badge](https://img.shields.io/badge/Efficientnet-grey) |
| [Multi-layer Perceptron from Scratch with Numpy](#multi-layer-perceptron-from-scratch) | ![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white) <br>![Static Badge](https://img.shields.io/badge/MLP-grey) ![Static Badge](https://img.shields.io/badge/Numpy-grey) |
| [Image Classification with BloodMNIST](#image-classification-with-bloodmnist-dataset) | ![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white) ![Keras](https://img.shields.io/badge/-Keras-FF0000?style=flat&logo=keras&logoColor=white) ![Tableau](https://img.shields.io/badge/-Tableau-FF0000?style=flat&logo=tableau&logoColor=white) <br> ![Static Badge](https://img.shields.io/badge/FCNN-grey) ![Static Badge](https://img.shields.io/badge/CNN-grey) ![Static Badge](https://img.shields.io/badge/Random_Forest-grey) ![Static Badge](https://img.shields.io/badge/SVM-grey) |
| [Delinquent Debtor Identification with R](#understanding-debtor-profiles) | ![R](https://img.shields.io/badge/-R_Studio-blue?style=flat&logo=r&logoColor=white) <br>![Static Badge](https://img.shields.io/badge/Random_Forest-grey) ![Static Badge](https://img.shields.io/badge/Logistic_Regression-grey) ![Static Badge](https://img.shields.io/badge/LDA-grey) ![Static Badge](https://img.shields.io/badge/AdaBoost-grey) ![Static Badge](https://img.shields.io/badge/SVM-grey) |


## ⭐ No-code Exploratory Data Analysis Platform


> <b> WORK IN PROGRESS (BEGINNING 27 Nov 2024). REPO CAN BE ACCESSED  <a href="https://github.com/vyun8699/monkey_business">[HERE]</a></b>

<b> Problem</b>: All data science project begins with a proper Exploratory Data Analysis (EDA) which can be repetitive and time consuming. Wouldn't it be nice if there is a tool that allow you to do simply click through, records all changes and output you've made a long the way, and gets you (mostly) there without coding?

<b> Solution</b>: We are creating a no-code platform which handles pre-processing semi-autonomously. The final output will be coded in node.js to make it pretty, and partially automated to save user's time.  

<b> Progress</b>: Some screenshot of current functionalities in Streamlit shown below. 

<p align="center">
<b>(Left)</b> File upload and automatic problem detection, <b>(Right)</b> Check parameter distribution <br>
<img src="assets/EDA_input.gif" width="48%"/> 
  &nbsp;&nbsp;&nbsp;
<img src="assets/EDA_distribution.gif" width="48%" />
</p>
 
<p align="center">
<br><b>(Left)</b> Check and delete null & duplicates, <b>(Right)</b> Transform parameters <br>
<img src="assets/EDA_null.gif" width="48%"/> 
  &nbsp;&nbsp;&nbsp; 
<img src="assets/EDA_transform.gif" width="48%" />
</p>


<p align="center">
Save output CSV and inspect change logs <br>
<img src="assets/EDA_log.gif" width="48%"/> 
</p>

## ⭐ Revenue optimization for an Australian commodity producer

> <b> ACTIVE NDA - REPOSITORY IS PRIVATE</b>

<b>Problem</b>: This project is done together with a boutique consulting firm for an Australian commodity producer. The client produces c.10,000 tons of raw output per year with 10 different quality metrics measured for individual batches. There are various finished products, each with their own quality requirements and market prices, which swings based on supply-demand.The client does not have a system to identify what product to produce with their existing stock at any given moment, which creates a habit of over-delivering on product requirements. In other words, the raw output used in any given product are often of too high quality, leaving gross margins on the table. 

<b>Solution</b>: Breadth first search (BFS) and Mixed Integer Linear Programming (MILP) were explored. The final optimization algorithm sits somewhere between BFS and MILP by reducing the search space up-front and returning a descending list of possible batch combinations based on their potential gross margin. 

<b> Impact </b>: A$15m of additional gross margins for a full year identified.  The solution will be hosted on GCP with a Streamlit overlay. This allows site managers to schedule combination reports before they start their day, on-the cloud, with negligible infrastructure cost. Site managers can also produce custom reports when necessary.


## ⭐ Small object detection model for an Australian drone company

><b> ACTIVE NDA - REPOSITORY IS PRIVATE</b>

<b>Problem</b>: This project aims to implement an automated object detection system to detect small & distant object in the horizon. The model will be ran locally on the small computer on-board the drone. The system is expected to function in all weather conditions, given enough natural light. 

<b> Solution</b> can be divided into three main parts: 
<ol> 
<li><b>Custom dataset</b>: built on several open-source datasets, controlled for object size distribution, annotation quality, image resolution, and diversity of weather/lighting condition.
<li> <b>Optimized model</b>: Several model architectures were fine-tuned (YOLOv8, YOLOv9, Faster RCNN, SSD, and SAM). Hyperparameter tuning done on YOLOv8-n/s/m. 
<li> <b>Slicing Aided Hyper Inferencing (SAHI)</b>: Implemented at inferencing to further aid the models accuracy in detecting small objects. 
</ol>

<b>Impact</b>: SAHI was particularly helpful with achieving the project objective. Learning from this project is used by the client in their on-board object detection system.


<p align="center">
  Detection of small objects on YOLOV8-m <b>(Left)</b> without SAHI and <b>(Right)</b> with SAHI 
  <img src="assets/SAHI_sample.png" height ="200">
</p>

<p align="center">
Small object detection with SAHI <br>
<img src="assets/yolov8_infer_sahi.gif" width="95%"/> 
</p>

## ⭐ Multi-class object classification via transfer learning

><b> REPO CAN BE ACCESSED <a href="https://github.com/vyun8699/CNN-via-transfer-learning">[HERE]</a> </b></br>
<b> TABLEAU DASHBOARD CAN BE ACCESSED <a href="https://public.tableau.com/app/profile/vincent.yunansan/viz/log_analyzer/Dashboard3?publish=yes">[HERE]</a> </b></br>

<b> Problem</b>: This project aims at fine-tuning open-source models to solve a multi-class classification problem. Training dataset consists of 30,000 images with 18 classes. 

<b> Solution</b>: Multiple pre-trained models from Pytorch were used. Softmax layer was adjusted to account for the multi-class classification task. Training done in batches and noise introduced with Dataloaders.  

<b>Impact</b>: The resulting model yielded 90%+ test F1 score with 5-hour traning runtime.

<p align="center">
  <img src="assets/regnetfeaturemap.png" height ="300">
  <br>
  Sample Feature Map Representation of RegNet
</p>



## ⭐ Multi-layer perceptron from scratch with Numpy

><b> REPO CAN BE ACCESSED <a href="https://github.com/vyun8699/MLP-from-scratch"> [HERE]</a></b>

<b> Problem</b>: Machine learning libraries often pack multiple mechanisms into a single function, which makes it hard to discern what happens underneath. 

<b> Solution</b>: This project uses Numpy to create a Multi Layer Perceptron (MLP) model. The system is split into three classes: 
<ol> 
<li> <b>Activation</b>: ReLu, Sigmoid, Tanh and their derivatives. </li>
<li> <b>Hidden Layer</b>: weights & biases, transformations such as batch normalization, momentum, dropout, weight decay, etc. </li>
<li> <b>Model</b>: batch training, loss functions, etfc. </li> </ol> 

Multiple parameters were tested against the training dataset. 
Summary below: 

<p align="center">
  <b> (Left - Right) </b> Experiment stages <br>
  <b> (Highlight) </b> Parameters in hyperparameter tuning <br>
  <img src="assets/MLP_appendix.png" height ="300">
</p>

## ⭐ Image Classification with BloodMNIST dataset

><b> REPO CAN BE ACCESSED <a href="https://github.com/vyun8699/BloodMNIST_classification"> [HERE] </a></b>

<b> Problem </b>: This study compares FCNN, CNN, Random Forest, and Support Vector Machine for blood cell image classification. The dataset used contain 17,000+ images of blood cells, resized to 28x28 pixels. The images are split into 8 classes as shown below:

<p align="center">
  <img src="assets/BloodMNIST_class.png" height ="500">
  <br>
  Description of classes in BloodMNIST   
  <br>
</p>

<b> Results </b>: 
<ol>
<li>Pre-processing was applied to each model as appropriate. 
<li>Hyperparameter tuning was done on number of neurons, activation function, learning rate, optimizer, regularizer, etc.  
<li> CNN was identified as best model. Please see report for details.
</ol>

<br>
<p align="center">
<b>(Left)</b> Pre-processing, <b>(Right)</b> Results <br>
<img src="assets/BloodMNIST_preprop.png" width="48%"/> 
  &nbsp;&nbsp;&nbsp;
<img src="assets/BloodMNIST_result.png" width="46%" />
</p>


## [⭐ Delinquent Debtor Identification with R](https://vyun8699.github.io/)

><b> REPO CAN BE ACCESSED <a href="https://vyun8699.github.io/"> [HERE]</a> </font></b>

<b> Problem </b>: This project identifies potential delinquent debtors based on their features in their initial debt application.

 <b>Solution</b>: This project was implemented in R. 
 <ol>
 <li> The following features were removed: features with cross-correlation, redundant features, forward-looking features. 
 <li>SMOTE was applied to improve data balance. 
<li> Pre-processing: principal component analysis, min-max scaling, one-hot-encoding. 
<li> Classification methods: 5 techniques were applied to find a method with <b> high True Positives</b> and <b> Low False Positives & Negatives </b></li>
</ol> 

<b>Results</b>: random forest was superior as measured by precision and sensitivity. Top-10 most important features identified by Random Forest shown below:
 

<br>
<p align="center">
<b>(Left)</b> Comparison scores, <b>(Right)</b> Top-10 most important features <br>
<img src="assets/R_output.png" height = '130' width="48%"/> 
  &nbsp;&nbsp;&nbsp;
<img src="assets/R_RFoutput.png" height = '130' width="48%" />
</p>

