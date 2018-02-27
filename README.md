# Video Surveillance for Road Traffic Monitoring

The goal of this project is to learn the basic concepts and techniques related to video sequences and apply them for surveillance applications

## Team 5:

| Name | E-mail | GitHub |
| :---: | :---: | :---: |
| [Lidia Garrucho](https://www.linkedin.com/in/lidia-garrucho-moras-77961a8a/) | lidiaxu.3@gmail.com | [LidiaGarrucho](https://github.com/LidiaGarrucho) |
| [Anna Martí](https://www.linkedin.com/in/annamartiaguilera/) | annamartiaguilera@gmail.com | [amartia](https://github.com/amartia) |
| [Santi Puch](https://www.linkedin.com/in/santipuch/) | santiago.puch.giner@gmail.com | [santipuch590](https://github.com/santipuch590) |
| [Xènia Salinas](https://www.linkedin.com/in/x%C3%A8nia-salinas-ventall%C3%B3-509081156/) | salinasxenia@gmail.com | [XeniaSalinas](https://github.com/XeniaSalinas) |

## Milestones 

### Week 1 - Introduction
1. Segmentation metrics. Understand precision & recall.
- [x] Implement Precision and Recall
- [x] Implement F1-Score
- [x] Compute these metrics for sequences A & B
- [x] Provide an interpretation of the different values obtained for Test A and Test B.

2. Temporal Analysis of the Results
- [x] Graph 1: True Positive & Total Foreground pixels vs #Frames.
- [x] Graph 2: F1-Score vs Time.
- [x] Provide an interpretation of the graphs obtained for Test A and Test B.

3. Quantitative evaluation of Optical Flow
- [x] Compute Mean Squared Error in Non-occluded areas (MSEN)
- [x] Compute Percentage of Erroneous Pixels in Non-occluded areas (PEPN)
- [x] Analysis and visualization of errors

4. Desynchronization of the sequence
- [x] Force de-synchronized results for background subtraction in Highway sequence
- [x] Study and comment the results

5. Optical Flow visualization
- [x] Propose a simplification method for a clean visualization

### Week 2 - Background estimation
[TO BE UPDATED]


## How to run the code:

For tasks 1, 2, and 4:

- `video_surveillance.py -c config/highway.py -t test_A`

- `video_surveillance.py -c config/highway.py -t test_B`

For tasks 3 and 5:

[**TODO:** Add how to run experiments using kitti dataset....]

In order to run the code, the datasets must be on the path *code/datasets/*.

The datasets needed to run the code and their folder organization are the following:

| ChangeDetection Dataset | Description | 
| :---: | :---: | 
| *highway/input/* |  ChangeDetection dataset input images | 
| *highway/groundtruth/*  |  ChangeDetection dataset ground truth | 
| *highway/results_testAB_changedetection/* |  ChangeDetection results using parameters A and B |

| Kitti Dataset | Description | 
| :---: | :---: | 
| *kitti//image_0/*|  Kitti dataset input images | 
| *kitti/flow_noc/*|  Kitti dataset ground truth | 
| *kitti/results/*|  Kitti dataset results | 

[TODO: (Proposal) Add more tables for every dataset used]
