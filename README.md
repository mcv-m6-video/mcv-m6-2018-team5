# Road Traffic Monitoring System

The goal of this project is to implement a Video Surveillance System for Traffic monitoring,  that allows to monitor the current state of the roads, being this the traffic density, mean speed and some other features of interest.

## System overview

<img align="left" width="370" height="300" src="https://github.com/mcv-m6-video/mcv-m6-2018-team5/blob/master/System_overview.PNG">

The first block of the system consists in a **Background Subtraction** algorithm based on adaptive Gaussian model defined in RGB space. On this step, we separate the vehicles from the environment.

The second block is a **Foreground Improvement** algorithm. The mask obtained in the previous step is improved with a morphology pipeline composed by a hole-filling step, area filtering, an opening and finally a closing.

The third block is a **Vehicle Tracker** based on Kalman Filter. The previous steps are executed for each frame independently; with this step, we obtain the path that follows each vehicle along the different frames.

Finally, the fourth block performs different **Road Statistics**. It estimates the speed of each detected vehicle and checks if it is below the speed limit of the road and some statistics per lane. Specifically it computes the total number of vehicles on each lane, the density of vehicles and the average speed.


## Materials

[**Slides**](https://docs.google.com/presentation/d/e/2PACX-1vQDxBlWMZOfcwImQG1Ge9C6QdzOAURoWQMcF4wx6sxOSdvlpBduN4AF-CIoIpM2lTrNiGxtyf75M5sw/pub?start=false&loop=false&delayms=3000)

**Report**: Work in Progress 

## Running the code

Check the README file in the `run` folder for further details on how to prepare the running environment and how to 
run the code.

## Project milestones 

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
1. Gaussian Modelling
- [x] Create a Gaussian function to model each background pixel using the first 50% of the video sequence
- [x] Segment the foreground pixels for the second 50% of the video sequence
- [x] Evaluate the segmentation: Compute F1-score vs alpha curve
- [x] Evaluate the segmentation: Precision-Recall vs alpha curve, Area Under the Curve
2. Adaptive Modelling
- [x] Segment the foreground pixels for the second 50% of the video sequence and left background adapt
- [x] Optimize alpha and rho with grid search to maximize F1-score
- [x] Compare Gaussian Modelling vs. Adaptive Modelling for all three sequences using F1-score and Area Under the Curve
3. Compare with state-of-the-art implementations
- [x] Run different state-of-the-art Background Subtraction methods: e.g.: OpenCV BackgroundSubtractorMOG, BackgroundSubtractorMOG2, BackgroundSubtractorLSBP
- [x] Evaluate precision vs recall to comment which method (single Gaussian programmed by you or state-of-the-art) performs better
- [x] Evaluate the sequences than benefit more one algorithm and try to explain why
4. Color Sequences
- [x] Update the implementation to support color sequences: Decide which colorspace to use, number of Gaussians per pixel, etc.

### Week 3 - Foreground segmentation
1. Hole filling
- [x] Choose the best configuration from week 2
- [x] Post-process with hole filling (indicate used library and link to code)
- [x] Experiment with 4 and 8 connectivity
- [x] Compute AUC and gain for each video sequence
- [x] Provide qualitative interpretation
2. Area filtering
- [x] AUC vs number of pixels
- [x] arg max P (AUC) (indicate used library and link to code)
3. Additional morphological operations
- [x] Explore with other morphological filters to improve AUC for foreground pixels (indicate used library and link to code)
4. Shadow removal
- [x] Implement some shadow removal existing techniques
5. Improvements of this week
- [x] Compare the precision / recall curves
- [x] Update the AUC and compute the gain

### Week 4 - Optical Flow and Video Stabilization
1. Optical Flow
- [x] Optical Flow with Block Matching
- [x] Block Matching vs Other Techniques:
- [x] Try OpenCV Farneback implementation
- [x] Try FlowNet 2.0 implementation

2. Video Stabilization
- [x] Video Stabilization wih Block Matching
- [x] Block Matching Stabilization vs Other Techniques
- [x] Stabilize your own video

### Week 5 - Region tracking and Kalman filter
1. Vehicle tracker
- [x] Vehicle tracker with Kalman filter
- [x] Vehicle tracker with other tools
2. Speed estimator
- [x] Estimate the speed of the vehicles
3. Road statistics
- [x] Check the speed limit
- [x] Check the road density of vehicles
- [x] Compute the average speed per lane

## Contributors

| Name | E-mail | GitHub |
| :---: | :---: | :---: |
| [Lidia Garrucho](https://www.linkedin.com/in/lidia-garrucho-moras-77961a8a/) | lidiaxu.3@gmail.com | [LidiaGarrucho](https://github.com/LidiaGarrucho) |
| [Anna Martí](https://www.linkedin.com/in/annamartiaguilera/) | annamartiaguilera@gmail.com | [amartia](https://github.com/amartia) |
| [Santi Puch](https://www.linkedin.com/in/santipuch/) | santiago.puch.giner@gmail.com | [santipuch590](https://github.com/santipuch590) |
| [Xènia Salinas](https://www.linkedin.com/in/x%C3%A8nia-salinas-ventall%C3%B3-509081156/) | salinasxenia@gmail.com | [XeniaSalinas](https://github.com/XeniaSalinas) |
