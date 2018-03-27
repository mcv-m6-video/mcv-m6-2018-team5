# Dataset
dataset_name                 = 'traffic'                        # Dataset name
dataset_path                 = 'datasets/traffic/input'         # Dataset path
gt_path                      = 'datasets/traffic/groundtruth'   # Ground truth path
results_path                 = 'datasets/traffic/results'

# Input Images
nr_images                    = 100
first_image                  = '000950'     # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

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

# Tracking parameters
distance_threshold            = 5
max_frames_to_skip            = 10
max_trace_length              = 15
init_estimate_error           = [200, 25]
motion_model_noise            = [100, 25]
measurement_noise             = 100.0

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
