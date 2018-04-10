# Dataset
dataset_name                 = 'relaxing_traffic'                        # Dataset name
dataset_path                 = 'datasets/relaxing_traffic/input'         # Dataset path
gt_path                      = 'datasets/relaxing_traffic/gt'            # Ground truth path
results_path                 = 'datasets/relaxing_traffic/results'
input_prefix                 = 'frame_'

# Input Images
nr_images                    = 1500
first_image                  = '0001'         # Fist image filename
image_type                   = 'jpg'            # Input image type
gt_image_type                = 'png'            # Ground truth image type
result_image_type            = 'jpg'
# Background images
first_back                    = '1405'
nr_back                       = 97

# Background Modelling
alpha                         = 3.7627
rho                           = 0.1578
modelling_method              = 'adaptive'      # adaptive, non-adaptive
color_space                   = "RGB"           # RGB, HSV

# Foreground Modelling
four_connectivity             = False
opening_strel                 = 'square'
opening_strel_size            = 5
closing_strel                 = 'square'
closing_strel_size            = 1
area_filtering                = True
area_filtering_P              = 400

# Kalman Filter
init_estimate_error           = [200, 25]
motion_model_noise            = [100, 25]
measurement_noise             = 100.0

# Multi-tracking parameters
cost_of_non_assignment        = 25
invisible_too_long            = 10
min_age_threshold             = 15

# Road statistics parameters
pixels_meter                  = [8.5, 7.5, 6.75, 8]
frames_second                 = 30
update_speed                  = 1
speed_estimate_running_avg    = 0.95
roi_speed                     = [
                                    [70, 220], [480, 220], [540, 275], [0, 275]
                                ]
lanes                         = [
                                    [[226, 102], [250, 102], [90, 295], [0, 280]],
                                    [[251, 102], [273, 102], [197, 295], [91, 295]],
                                    [[305, 102], [324, 102], [453, 295], [351, 295]],
                                    [[325, 102], [343, 102], [563, 294], [454, 294]],
                                ]
max_speed                     = 120
high_density                  = 5

#Visualization
margin = 150

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
