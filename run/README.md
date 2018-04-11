## Development environment

This project has been developed and tested with **Python 2.7**.

### Using `conda`

The preferred method to prepare the development environment is using `conda`, as the whole environment is easily 
reproducible for the same platform:

If you are running in a **Linux 64** platform, run the following command:
```
$ conda create --name va_project_t5 --file spec-file-linux-64.txt
```
Alternatively, if you are runnign in a **Windows 64** platform, you should run:
```
$ conda create --name va_project_t5 --file spec-file-win-64.txt
```

Once the creation process finishes you can `activate` the environment via `$ source activate va_project_t5`.

### Using `pip`

If you are using a `virtualenv` or the system Python, you can use `pip` to install the required libraries to run this 
code:
```
$ pip install -r requirements.txt
```

Unlike with `conda`, you will have to install **OpenCV** either by downloading the pre-built binaries for your platform or compiling it from 
source.


## How to run the code

### Week 1

For tasks 1, 2, and 4:

- `python cli/metrics_evaluation.py --config-path config/highway_evaluate.py --test-name test_A`

- `python cli/metrics_evaluation.py --config-path config/highway_evaluate.py --test-name test_B`

For tasks 3 and 5:

- `python cli/metrics_evaluation.py --config-path config/kitti_evaluate.py --test-name test_flow`

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

- `python cli/background_modeling.py --config-path config/highway_background.py --test-name highway`
- `python cli/background_modeling.py --config-path config/traffic_background.py --test-name traffic`
- `python cli/background_modeling.py --config-path config/fall_background.py --test-name fall`

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

- `python cli/foreground_estimation.py --config-path config/highway_background.py --test-name highway`
- `python cli/foreground_estimation.py --config-path config/traffic_background.py --test-name traffic`
- `python cli/foreground_estimation.py --config-path config/fall_background.py --test-name fall`

The datasets needed to run the code and their folder organization are the following:

| ChangeDetection Dataset | Description |
| :---: | :---: |
| *highway/input/* |  Highway sequence (baseline) input images |
| *highway/groundtruth/*  |  Highway sequence (baseline) ground truth |
| *traffic/input/* |  Traffic sequence (camera jitter) input images |
| *traffic/groundtruth/*  |  Traffic sequence (camera jitter) ground truth |
| *fall/input/* |  Fall sequence (adaptive background) input images |
| *fall/groundtruth/*  |  Fall sequence (adaptive background) ground truth |


### Week 4

For all the tasks, for each of the sequences:

- `python cli/optical_flow_estimation.py --config-path config/kitti_evaluate.py --test-name kitti`

The datasets needed to run the code and their folder organization are the following:

| Kitti Dataset | Description |
| :---: | :---: |
| *kitti//image_0/*|  Kitti dataset input images |
| *kitti/flow_noc/*|  Kitti dataset ground truth |
| *kitti/results/*|  Kitti dataset results |


### Week 5

For task 1 with Kalman filter, for each of the sequences:

- `python cli/vehicle_tracker.py --config-path config/highway_tracker.py --test-name highway`
- `python cli/vehicle_tracker.py --config-path config/traffic_tracker.py --test-name traffic`

For task 1 with other tools, for each of the sequences:

- `python cli/vehicle_tracker_sota.py --config-path config/highway_tracker.py --test-name highway`
- `python cli/vehicle_tracker_sota.py --config-path config/traffic_tracker.py --test-name traffic`

For task 2 using homography, for each of the sequences:

- `python cli/speed_estimator.py --config-path config/highway_speed.py --test-name highway`
- `python cli/speed_estimator.py --config-path config/traffic_speed.py --test-name traffic`

For task 3, for each of the sequences:

- `python cli/road_statistics.py --config-path config/highway_statistics.py --test-name highway`
- `python cli/road_statistics.py --config-path config/traffic_statistics.py --test-name traffic`
- `python cli/road_statistics.py --config-path config/alibi_statistics.py --test-name alibi`
- `python cli/road_statistics.py --config-path config/relaxing_traffic_statistics.py --test-name relaxing_traffic`
- `python cli/road_statistics.py --config-path config/street_statistics.py --test-name street`