# Dataset
dataset_name                 = 'traffic'                        # Dataset name
dataset_path                 = 'datasets/traffic/input'         # Dataset path
gt_path                      = 'datasets/traffic/groundtruth'   # Ground truth path
results_path                 = 'datasets/traffic/results'

# Input Images
nr_images                    = 1570
first_image                  = '000001'     # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Background Modelling
alpha                         = 3.7627
rho                           = 0.1578
first_back                    = '000015'
nr_back                       = 13
modelling_method              = 'adaptive'      # adaptive, non-adaptive
color_space                   = "RGB"           # RGB, HSV

# Foreground Modelling
four_connectivity             = False
opening_strel                 = 'diagonal'
opening_strel_size            = 10
closing_strel                 = 'diamond'
closing_strel_size            = 10
area_filtering                = True
area_filtering_P              = 820

# Kalman Filter
init_estimate_error           = [200, 25]
motion_model_noise            = [100, 25]
measurement_noise             = 100.0

# Multi-tracking parameters
cost_of_non_assignment        = 50
invisible_too_long            = 7
min_age_threshold             = 15

# Road statistics parameters
pixels_meter                  = 8.89
frames_second                 = 25.0
update_speed                  = 1
roi_speed                     = [[5, 64], [154, 6], [309, 103], [125, 226]]
lanes                         = [[[0, 0], [45, 0], [311, 238], [138, 238], [4, 56]], [[58, 0], [152, 0], [317, 100], [317, 222]]]
max_speed                     = 100

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
