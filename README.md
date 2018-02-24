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
- [ ] Provide an interpretation of the different values obtained for Test A and Test B.

2. Temporal Analysis of the Results
- [x] Graph 1: True Positive & Total Foreground pixels vs #Frames.
- [x] Graph 2: F1-Score vs Time.
- [ ] Provide an interpretation of the graphs obtained for Test A and Test B.

3. Quantitative evaluation of Optical Flow
- [ ] Compute Mean Squared Error in Non-occluded areas (MSEN)
- [ ] Compute Percentage of Erroneous Pixels in Non-occluded areas (PEPN)

4. Desynchronization of the Results
- [x] Force de-synchronized results for background substraction in Highway sequence.
- [ ] Study and comment the results

5. Optical Flow visualization
- [ ] Propose a simplification method for a clean visualization

### Week 2 - Background estimation
[TO BE UPDATED]


## How to run the code:

[**TODO:** Add info about paths, how to run experiments, ....]
Task 1. Evaluate sequences A and B
- `video_surveillance.py -c config/highway -t test_A`

- `video_surveillance.py -c config/highway -t test_B`

