# Dataset
dataset_name                 = 'traffic'                        # Dataset name
dataset_path                 = 'datasets/alibi/input'         # Dataset path
input_prefix                 = 'alibi_ip_camera_'
gt_path                      = ''
results_path                 = 'datasets/alibi/results'

# Input Images
nr_images                    = 1171
first_image                  = '0403'     # Fist image filename
image_type                   = 'jpg'        # Input image type
result_image_type            = 'png'

# Background Modelling
alpha                         = 3.7627
rho                           = 0.1578
first_back                    = '1428'
nr_back                       = 19
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
roi_speed                     = [[84, 189], [294, 189], [339, 242], [46, 242]]
lanes                         = [[[140, 120], [174, 120], [133, 253], [38, 253]], [[182, 120], [208, 120], [238, 253], [147, 253]], [[219, 120], [246, 120], [343, 253], [253, 253]]]
max_speed                     = 100

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
