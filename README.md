# 🔭 Vincent Yunansan, MBA CFA 
<div style="text-align: justify;">

<div style="border: 1px solid #ccc; padding: 10px;">
<b>Welcome to my GitHub profile!</b>

<br>
I'm a former investment banking and private equity professional with over 7 years of experience who has taken the leap into the world of data science. I took the Master's in Data Science program at USYD to explore different problem sets in analytics and machine learning. My unique background allows me to bridge the gap between business needs and technical solutions, effectively managing stakeholders while delivering data-driven insights. I'm passionate about leveraging my diverse skill set to tackle complex challenges and drive innovation in the field of data science. 
</br>

<br> My latest CV can be accessed here: <a href="assets/Resume_Vincent_Yunansan.pdf"> Curriculum Vitae</a> </br>


Have any questions? Below is my contact information:
- Email &ensp;&ensp;: vincent.yunansan@gmail.com  
- Phone &ensp;: +61459961345
</br>

</div>
<br>

# Project Showcase

## 🌾 Production optimization for an Australian commodity producer
![Static Badge](https://img.shields.io/badge/Python-green)
![Static Badge](https://img.shields.io/badge/breadth_first_search-blue)
![Static Badge](https://img.shields.io/badge/MILP-red)
![Static Badge](https://img.shields.io/badge/GCP-yellow)


<details>
<summary> Expand for details
</summary>
<br> <i> <font color = "grey">Due to active non-disclosure agreement, details of this project is not made available in this repository. Details provided has been masked to protect confidentiality</i> </font> </br>
<br> <b> Problem </b>: This project aims to implement a recommendation system to find combination of batches with highest market prices. The proposed combination has to satisfy 10 different quality metrics. These metrics can be improved by machine processes at a cost of lower output yields. The client has to process c.10,000 batches a year and react to price swings, machine down-time, etc.</br>
<br><b> Solution </b>: Breadth first search (BFS) and Mixed Integer Linear Programming (MILP) were explored. The final optimization method sits somewhere between BFS and MILP by taking out combinations that are too expensive or impossible to produce, search for combinations in the remaining search space, and returns a list of possible batch combinations (without duplicates) in a descending list. </br> 

<br><b> Implementation </b>: The solution is hosted on GCP with a Streamlit overlay. This allows site managers to schedule combination reports before they start their day, on-the cloud, with negligible infrastructure cost. Site managers can also produce custom reports when necessary.
</br>

</details>

## 🚁 Small object detection model for an Australian drone company
![Static Badge](https://img.shields.io/badge/Python-green)
![Static Badge](https://img.shields.io/badge/Pytorch-orange)
![Static Badge](https://img.shields.io/badge/YOLO-blue)
![Static Badge](https://img.shields.io/badge/SSD-blue)
![Static Badge](https://img.shields.io/badge/FRCNN-blue)
![Static Badge](https://img.shields.io/badge/SAM-blue)
![Static Badge](https://img.shields.io/badge/Edge_hardware-yellow)


<details>
<summary> Expand for details
</summary>
<br> <i> <font color = "grey">This is my Capstone Project for the MDS project at USYD. Due to active non-disclosure agreement, details of this project is not made available in this repository. Details provided has been masked to protect confidentiality.</i> </font> </br>
<br> <b> Problem </b>: This project aims to implement an automated object detection system able to detect small distant object in outdoor conditions, to be installed on a small computer on-board the vehicle.</br>
<br><b> This project is on-going. The solution can be divided into three branches </b>:   
<ol> 
<li> </b> Custom dataset</b> built on open-source datasets which has significant sample of small objects of interest. </li>
<li> </b> Optimized model</b> from multiple computer vision model architectures, including YOLO, Faster RCNN, SSD, and SAM. </li>
<li> </b> Hardware implementation </b> where the chosen model has to be able to infer rapidly on-board the vehicle. </li>
</br> 



</details>

