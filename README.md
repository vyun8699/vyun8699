# 🔭 Vincent Yunansan, MBA CFA 
<div style="text-align: justify;">

<div style="border: 1px solid #ccc; padding: 10px;">
<b>Hi there!</b>
<br>
<br>
I'm a former investment banking and private equity professional with over 7 years of experience who has taken the leap into the world of data science. 
I took the Master's in Data Science program at USYD as a career break to explore different problem sets in analytics and machine learning. 
</br>
  <br>
I am most interested in building repeatable/scalable solutions to process large amounts of data while maintaining simplicity. 
</br>

<br><b> What I'm working on right now </b>: 
- No-code Exploratory Data Analysis Platform
- Template of customizable RAG framework to be fully hosted on GCP

<br> My latest CV can be accessed here: <a href="assets/Resume_Vincent_Yunansan.pdf"> Curriculum Vitae</a> </br>


Have any questions? Below is my contact information:
- Email &ensp;&ensp;: vincent.yunansan@gmail.com  
- Phone &ensp;: +61459961345
</br>

</div>
<br>


## Project Showcase - Table of Contents
1. [No-code Exploratory Data Analysis Platform](#no-code-exploratory-data-analysis-platform)
2. [Revenue Optimization for Australian Commodity Producer](#revenue-optimization-for-an-australian-commodity-producer)
3. [Small Object Detection Model](#small-object-detection-model-for-an-australian-drone-company)
4. [Multi-class Object Classification](#multi-class-object-classification-via-transfer-learning)
5. [Multi-layer Perceptron from Scratch](#multi-layer-perceptron-from-scratch)
6. [Image Classification with BloodMNIST](#image-classification-with-bloodmnist-dataset)
7. [Understanding Debtor Profiles](#understanding-debtor-profiles)

## No-code Exploratory Data Analysis Platform
![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white)  ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

<br> <i> This project began on 27 Nov 2024. Repository can be accessed  <a href="https://github.com/vyun8699/monkey_business">here.</a></i></br>

<br> <b> Problem </b>: All data science project begins with a proper Exploratory Data Analysis (EDA) which can be repetitive and time consuming. Wouldn't it be nice if there is a tool that allow you to do simply click through, records all changes and output you've made a long the way, and gets you (mostly) there without coding?</br>
<br><b> Solution </b>: This is a work in progress. The final output will be coded in node.js to make it pretty, and partially automated to save user's time.  </br> 

</br>



<div style="display: flex; justify-content: space-between; align-items: center; gap:10px">
  <div style="text-align: center;">
    <img src="assets/EDA_input.gif" style="height: 200px;">
    <br>
    1. File upload and automatic problem detection
  </div>
  <div style="text-align: center;">
    <img src="assets/EDA_distribution.gif" style="height: 200px;">
    <br>
    2. Check parameter distribution
  </div>
</div>

<div style="display: flex; justify-content: space-between; align-items: center; gap:10px">
  <div style="text-align: center;">
    <img src="assets/EDA_null.gif" style="height: 200px;">
    <br>
    3. Check null & duplicates (and delete if needed)
  </div>
  <div style="text-align: center;">
    <img src="assets/EDA_transform.gif" style="height: 200px;">
    <br>
    4. File upload and initial inspection
  </div>
</div>

<div style="display: flex; justify-content: space-between; align-items: center; gap:20px">
  <div style="text-align: center;">
    <img src="assets/EDA_log.gif" style="height: 200px;">
    <br>
    5. Save output CSV and log
  </div>
</div>

## Revenue optimization for an Australian commodity producer
![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white)
![GCP](https://img.shields.io/badge/-GCP-yellow?style=flat&logo=Google&logoColor=white)
![Static Badge](https://img.shields.io/badge/breadth_first_search-grey)
![Static Badge](https://img.shields.io/badge/MILP-grey)

<br> <i> Due to active non-disclosure agreement, details of this project is not made available in this repository. </i> </br>

This project is done in tandem with a boutique consulting firm for an Australian commodity producer. The client produces c.10,000 tons of raw output per year with 10 different quality metrics measured for individual batches. There are various finished products, each with their own market prices and quality specifications.

<b>Problem</b>: The client does not have a system to identify what product to produce with their existing stock at any given moment, which creates a habit of over-delivering on product requirements. In other words, the raw output used in any given product are often of too high quality, leaving gross margins on the table. 

<b>Output</b>: Breadth first search (BFS) and Mixed Integer Linear Programming (MILP) were explored. The final optimization algorithm sits somewhere between BFS and MILP by ignoring combinations that are too expensive or impossible to produce, searching for combinations in the remaining search space, and returning a list of possible batch combinations (without duplicates) in a descending list.

<b> Implementation </b>: The solution will be hosted on GCP with a Streamlit overlay. This allows site managers to schedule combination reports before they start their day, on-the cloud, with negligible infrastructure cost. Site managers can also produce custom reports when necessary.


## Small object detection model for an Australian drone company
![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white)
![Pytorch](https://img.shields.io/badge/-Pytorch-orange?style=flat&logo=pytorch&logoColor=white)
![Pytorch](https://img.shields.io/badge/-TensorFlow-red?style=flat&logo=tensorflow&logoColor=white)
![Static Badge](https://img.shields.io/badge/YOLO-grey)
![Static Badge](https://img.shields.io/badge/SSD-grey)
![Static Badge](https://img.shields.io/badge/FRCNN-grey)
![Static Badge](https://img.shields.io/badge/SAM-grey)
![Static Badge](https://img.shields.io/badge/Edge_hardware-white)

<br> <i> <font color = "grey">This is my Capstone Project for the MDS program at USYD. 
<br>Due to active non-disclosure agreement, details of this project is not made available in this repository.</i> </font> </br>
<br>This project aims to implement an automated object detection system able to detect small distant object in outdoor conditions, to be installed on a small computer on-board the vehicle. The proposed solution can be divided into three branches:

<li> <b>Custom dataset</b>: built on open-source datasets which has significant sample of small objects of interest. </li>
<li> <b>Optimized model</b>: from multiple computer vision model architectures, including YOLO, Faster RCNN, SSD, and SAM. </li>
<li> <b>Inferencing optimization</b>: we explored the use of SAHI to aid with small object detection in real time. </li> </ol> 
</br>

<p align="center">
  <img src="assets/SAHI_sample.png" height ="200">
  <br>
  Sample technique to solve the small object detection problem
</p>

## Multi-class object classification via transfer learning
![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white)
![Tableau](https://img.shields.io/badge/-Tableau-FF0000?style=flat&logo=tableau&logoColor=white)
![Pytorch](https://img.shields.io/badge/-Pytorch-orange?style=flat&logo=pytorch&logoColor=white)
![Pytorch](https://img.shields.io/badge/-Google_Colab-yellow?style=flat&logo=googlecolab&logoColor=white)
![Static Badge](https://img.shields.io/badge/GoogLeNet-grey)
![Static Badge](https://img.shields.io/badge/ResNext-grey)
![Static Badge](https://img.shields.io/badge/Shufflenet-grey)
![Static Badge](https://img.shields.io/badge/Efficientnet-grey)

<br> <i> Details of this project can be accessed <a href="https://github.com/vyun8699/CNN-via-transfer-learning">here</a> </i></br>

<br> <b> Problem </b>: Training a model from scratch requires massive computational resources not accessible to the common enthusiast. This project showcases methods to access open-source models and fine-tune them to solve a multi-class classification problem. </br>
<br><b> Solution & Implementation </b>:   
<ol> 
<li> <b>Dataset</b>: 30,000 images with 18 classes are split into train and validation sets. </li>
<li> <b>Dataloader</b>: images are loaded in batches to avoid bottlenecking. Transformations applied to increase model robustness. 
<li> <b>Models</b>: pre-trained models from Pytorch are customized to handle multi-class classification. Various methods are applied to aid training speeds and scores. Tableau is used to visualize training statistics.</li>
<li> <b>Performance</b>: micro F1, model size and training runtime are considered together to recommend the best system. </li>
<li> <b>Results</b>: The best model yielded 90%+ test F1 score with 5-hour training run-time. </ol> 

<b> Reflection </b>: computers do not perceive image data as humans do. The images below shows how an image of a cat travels through the layers in RegNet. We can see how the model is able to differentiate features of our object of interest (e.g. the cat) against the surrounding environment. 

<p align="center">
  <img src="assets/regnetfeaturemap.png" height ="500">
  <br>
  Sample Feature Map Representation of RegNet
</p>

## Multi-layer perceptron from scratch
![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white)
![Static Badge](https://img.shields.io/badge/MLP-grey)
![Static Badge](https://img.shields.io/badge/Numpy-grey)

<br> <i> <font color = "grey">Details of this project can be accessed <a href="https://github.com/vyun8699/MLP-from-scratch"> here</a> </font> </i></br>

<br> <b> Problem </b>: The goal of this project is to implement a neural network without the use of modern machine learning libraries. By doing so, we will showcase the effects of different methods/components to the overall quality of our MLP model on the provided dataset.</br>
<br><b> This project is implemented in Numpy to show how modern machine learning framework ingests data. The system is divided into several steps </b>:   
<ol> 
<li> Pre-processing: input is normalized for better ingestion by the MLP and split into train-validation-test sets. </li>
<li> Base architecture: the MLP is implemented in 3 classes for code hygiene. </li>
<li> Methods: batch training, early stopping, weight decay, dropout, momentum, batch normalization, adam. </li> </ol> 
</br>

<b> Reflection </b>: I am a firm believer of the iterative process. This project was built in multiple stages where different methods were explored and the best configuration was implemented into the core build. The iterative process is useful in exploring builds efficiently while maintaining explainability. 

<p align="center">
  <img src="assets/MLP_appendix.png" height ="300">
  <br>
  Experiment stages in the MLP from scratch project.   
  <br>Methods in blue are tested while keeping all else constant.
</p>

## Image Classification with BloodMNIST dataset
![Python](https://img.shields.io/badge/-Python-Green?style=flat&logo=python&logoColor=white)
![Tableau](https://img.shields.io/badge/-Keras-FF0000?style=flat&logo=keras&logoColor=white)
![Static Badge](https://img.shields.io/badge/FCNN-grey)
![Static Badge](https://img.shields.io/badge/CNN-grey)
![Static Badge](https://img.shields.io/badge/Random_Forest-grey)
![Static Badge](https://img.shields.io/badge/SVM-grey)

<br> <i> <font color = "grey">Report and workbook for this project can be accessed <a href="https://github.com/vyun8699/BloodMNIST_classification"> here</a> </font> </i></br>

<br> <b> Problem </b>: This study compares four machine learning methods in blood cell image classification. The methods are compared by their overall speed, performance scores, and error rates. 

<br> <b> Dataset </b>: 17,000+ images of blood cells, resized to 28x28 pixels for efficiency. The images are split into 8 classes as shown below:

<p align="center">
  <img src="assets/BloodMNIST_class.png" height ="500">
  <br>
  Description of classes in BloodMNIST   
  <br>
</p>

<br> <b> Pre-processing </b>: Different pre-processing steps are applied to each method as appropriate. Please see report for details.

<p align="center">
  <img src="assets/BloodMNIST_preprop.png" height ="350">
  <br>
  Pre-processing steps for each method   
  <br>
</p>

<br> <b> Methods </b>: Different parameters are tuned for each method, including number of neurons, activation function, learning rate, optimizer, regularizer, etc. Please see report for details. 

<br> <b> Results </b>: Best configuration for each method shown below, along with their class-specific scores. Convolutionary Neural Network (CNN) performs best amongst methods considered. 

<p align="center">
  <img src="assets/BloodMNIST_result.png" height ="300">
  <br>
  Tuning results
  <br>
</p>

<b> Reflection </b>: Each algorithm has different use cases and trade-offs which should be considered. The choice of algorithm must be matched with the goal of the exercise itself. More complex algorithm may take longer to run and may not provide the best explanation/reasoning on feature importance.  Users need to consider what output needs to be generated to (i) measure the quality of the model, and (ii) provide useful insights when designing a model protocol. Simpler models may provide adequate outputs with lower computational costs (e.g., shorter runtimes) given the right pre-processing and design.

## Understanding debtor profiles
![R](https://img.shields.io/badge/-R_Studio-blue?style=flat&logo=r&logoColor=white)
![Static Badge](https://img.shields.io/badge/Random_Forest-grey)
![Static Badge](https://img.shields.io/badge/Logistic_Regression-grey)
![Static Badge](https://img.shields.io/badge/LDA-grey)
![Static Badge](https://img.shields.io/badge/AdaBoost-grey)
![Static Badge](https://img.shields.io/badge/SVM-grey)

<br> <i> <font color = "grey">Report for this project can be accessed <a href="https://vyun8699.github.io/"> here</a> </font> </i></br>

<br> <b> Problem </b>: Understanding relationships between factors in demographic data is equally as important to having access to them in the first place. This project displays how bad debtors can be identified from application data by using different machine learning methods.</br>
<br> <b> Side note </b>: This analysis is done on a public dataset hosted on Kaggle. Many submissions on the platform claims 99% accuracy but most of them suffer from data leaks and cross correlation. This happens when the analyst does not do proper data exploration and implemented lines of code onto the problem.</br>

<br><b> This project is implemented in R Studio and follows the following steps </b>:   
<ol> 
<li> Data exploration and pre-processing: cross-correlation and data distribution are analyzed to allow for fair analysis. We removed cross-correlated features and any features that would leak forward-looking information. We applied SMOTE to alleviate data imbalance. </li>
<li> Pre-processing: principal component analysis, min-max scaling, one-hote-encoding, SMOTE. </li>
<li> Classification methods: 5 techniques were applied to find a method with <mark>high True Positives</mark> and <mark> Low False Positives & Negatives </mark></li>
<li> Results: random forest was superior as measured by Precision and Sensitivity. 
</ol> 
</br>

<p align="center">
  <img src="assets/R_output.png" height ="150">
  <br>
 Comparison scores
</p>

<p align="center">
  <img src="assets/R_RFoutput.png" height ="250">
  <br>
Top-10 most important features based on Random Forest
</p>
