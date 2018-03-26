# Dataset
dataset_name                 = 'highway'                        # Dataset name
dataset_path                 = 'datasets/highway/input'         # Dataset path
gt_path                      = 'datasets/highway/groundtruth'   # Ground truth path
results_path                 = 'datasets/highway/results/'

# Input Images
nr_images                    = 300
first_image                  = '001050'         # Fist image filename
image_type                   = 'jpg'            # Input image type
gt_image_type                = 'png'            # Ground truth image type
result_image_type            = 'png'

# Background Modelling
alpha                         = 3.051
rho                           = 0.211

modelling_method              = 'adaptive'      # adaptive, non-adaptive
color_space                   = "RGB"           # RGB, HSV

# Foreground Modelling
four_connectivity             = False
opening_strel                 = 'square'
opening_strel_size            = 5
closing_strel                 = 'diamond'
closing_strel_size            = 3
area_filtering                = True
area_filtering_P              = 128

# Tracking parameters
distance_threshold            = 5
max_frames_to_skip            = 10
max_trace_length              = 15

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
