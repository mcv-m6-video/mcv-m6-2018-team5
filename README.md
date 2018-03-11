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
- [ ] AUC vs number of pixels
- [ ] arg max P (AUC) (indicate used library and link to code)
3. Additional morphological operations
- [ ] Explore with other morphological filters to improve AUC for foreground pixels (indicate used library and link to code)
4. Shadow removal
- [ ] Implement some shadow removal existing techniques
5. Improvements of this week
- [ ] Compare the precision / recall curves
- [ ] Update the AUC and compute the gain

## How to run the code

### General information

The Command Line Interface (CLI) usage is the following:

```
$ python video_surveillance.py 
usage: video_surveillance.py [-h] -c CONFIG_PATH -t TEST_NAME
                             {evaluate_metrics,background_estimation}

```

It provides the following help information:
```
usage: video_surveillance.py [-h] -c CONFIG_PATH -t TEST_NAME
                             {evaluate_metrics,background_estimation}

Video surveillance application, Team 5

positional arguments:
  {evaluate_metrics,background_estimation}
                        Task to run

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_PATH, --config-path CONFIG_PATH
                        Configuration file path
  -t TEST_NAME, --test-name TEST_NAME
                        Name of the test
```

### Week 1

For tasks 1, 2, and 4:

- `python video_surveillance.py evaluate_metrics --config-path config/highway_evaluate.py --test-name test_A`

- `python video_surveillance.py evaluate_metrics --config-path config/highway_evaluate.py --test-name test_B`

For tasks 3 and 5:

- `python video_surveillance.py evaluate_metrics --config-path config/kitti_evaluate.py --test-name test_flow`

In order to run the code, the datasets must be on the path *datasets/*.

The datasets needed to run the code and their folder organization are the following:

| ChangeDetection Dataset | Description | 
| :---: | :---: | 
| *highway/input/* |  Highway sequence (baseline) input images | 
| *highway/groundtruth/*  |  Highway sequence (baseline) ground truth | 
| *highway/results_testAB_changedetection/* |  Highway sequence (baseline) results using parameters A and B |

| Kitti Dataset | Description | 
| :---: | :---: | 
| *kitti//image_0/*|  Kitti dataset input images | 
| *kitti/flow_noc/*|  Kitti dataset ground truth | 
| *kitti/results/*|  Kitti dataset results | 

### Week 2

For tasks 1 and 2, for each of the sequences:

- `python video_surveillance.py background_estimation --config-path config/highway_background.py --test-name highway`
- `python video_surveillance.py background_estimation --config-path config/traffic_background.py --test-name traffic`
- `python video_surveillance.py background_estimation --config-path config/fall_background.py --test-name fall`

The datasets needed to run the code and their folder organization are the following:

| ChangeDetection Dataset | Description | 
| :---: | :---: | 
| *highway/input/* |  Highway sequence (baseline) input images | 
| *highway/groundtruth/*  |  Highway sequence (baseline) ground truth | 
| *traffic/input/* |  Traffic sequence (camera jitter) input images | 
| *traffic/groundtruth/*  |  Traffic sequence (camera jitter) ground truth | 
| *fall/input/* |  Fall sequence (adaptive background) input images | 
| *fall/groundtruth/*  |  Fall sequence (adaptive background) ground truth | 


### Week 3

For all the tasks, for each of the sequences:

- `python video_surveillance.py foreground_estimation --config-path config/highway_background.py --test-name highway`
- `python video_surveillance.py foreground_estimation --config-path config/traffic_background.py --test-name traffic`
- `python video_surveillance.py foreground_estimation --config-path config/fall_background.py --test-name fall`

The datasets needed to run the code and their folder organization are the following:

| ChangeDetection Dataset | Description |
| :---: | :---: |
| *highway/input/* |  Highway sequence (baseline) input images |
| *highway/groundtruth/*  |  Highway sequence (baseline) ground truth |
| *traffic/input/* |  Traffic sequence (camera jitter) input images |
| *traffic/groundtruth/*  |  Traffic sequence (camera jitter) ground truth |
| *fall/input/* |  Fall sequence (adaptive background) input images |
| *fall/groundtruth/*  |  Fall sequence (adaptive background) ground truth |

**TODO**: UPDATE & COMPLETE THIS SECTION
