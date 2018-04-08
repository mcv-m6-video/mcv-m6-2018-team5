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
first_back                    = '1428'
nr_back                       = 19

# Background Modelling
alpha                         = 3.7627
rho                           = 0.1578
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
cost_of_non_assignment        = 10
invisible_too_long            = 10
min_age_threshold             = 15

# Road statistics parameters
pixels_meter                  = 8.89
frames_second                 = 30
update_speed                  = 1
roi_speed                     = [[84, 189], [294, 189], [339, 242], [46, 242]]
lanes                         = [[[140, 120], [174, 120], [133, 253], [38, 253]], [[182, 120], [208, 120], [238, 253], [147, 253]], [[219, 120], [246, 120], [343, 253], [253, 253]]]
max_speed                     = 100

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
