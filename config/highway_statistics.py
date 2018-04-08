# Dataset
dataset_name                 = 'highway'                        # Dataset name
dataset_path                 = 'datasets/highway/input'         # Dataset path
input_prefix                 = 'in'
gt_path                      = 'datasets/highway/groundtruth'   # Ground truth path
results_path                 = 'datasets/highway/results'

# Input Images
nr_images                    = 1700
first_image                  = '000001'         # Fist image filename
image_type                   = 'jpg'            # Input image type
gt_image_type                = 'png'            # Ground truth image type
result_image_type            = 'png'
# Background images
first_back                    = '000469'
nr_back                       = 25

# Background Modelling
alpha                         = 3.051
rho                           = 0.211
modelling_method              = 'adaptive'      # adaptive, non-adaptive
color_space                   = "RGB"           # RGB, HSV

# Foreground Modelling
four_connectivity             = False
opening_strel                 = 'square'
opening_strel_size            = 5
closing_strel                 = 'square'
closing_strel_size            = 10
area_filtering                = True
area_filtering_P              = 128

# Kalman Filter
init_estimate_error           = [200, 25]
motion_model_noise            = [100, 25]
measurement_noise             = 100.0

# Multi-tracking parameters
cost_of_non_assignment        = 10
invisible_too_long            = 7
min_age_threshold             = 15

# Speed parameters
pixels_meter                  = 4.97
frames_second                 = 25.0
update_speed                  = 1
roi_speed                     = [[104, 119], [262, 119], [252, 222], [7, 222]]
lanes                         = [[[205, 12], [235, 12], [112, 237], [5, 237]], [[243, 12], [267, 12], [245, 237], [127, 237]]]
max_speed                     = 80

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
