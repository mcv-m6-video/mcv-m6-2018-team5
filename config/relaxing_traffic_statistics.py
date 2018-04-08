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
result_image_type            = 'png'
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
opening_strel_size            = 10
closing_strel                 = 'square'
closing_strel_size            = 10
area_filtering                = True
area_filtering_P              = 100

# Kalman Filter
init_estimate_error           = [200, 25]
motion_model_noise            = [100, 25]
measurement_noise             = 100.0

# Multi-tracking parameters
cost_of_non_assignment        = 10
invisible_too_long            = 10
min_age_threshold             = 15

# Road statistics parameters
pixels_meter                  = 8.89
frames_second                 = 30
update_speed                  = 1
roi_speed                     = [
                                    [0, 280], [226, 102], [343, 102], [479, 294],
                                ]
lanes                         = [
                                    [[0, 280], [226, 102], [250, 102], [90, 295]],
                                    [[91, 295], [251, 102], [273, 102], [197, 295]],
                                    [[351, 295], [305, 102], [324, 102], [453, 295]],
                                    [[454, 294], [325, 102], [343, 102], [479, 294]],
                                ]
max_speed                     = 120

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
