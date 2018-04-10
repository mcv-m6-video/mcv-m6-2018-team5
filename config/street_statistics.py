# Dataset
dataset_name                 = 'street_light'                        # Dataset name
dataset_path                 = 'datasets/street_light/input'         # Dataset path
input_prefix                 = 'in'
gt_path                      = ''   # Ground truth path
results_path                 = 'datasets/street_light/results'

# Input Images
nr_images                    = 3200
first_image                  = '000001'         # Fist image filename
image_type                   = 'jpg'            # Input image type
gt_image_type                = 'png'            # Ground truth image type
result_image_type            = 'png'
# Background images
first_back                    = '000111'
nr_back                       = 11

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
pixels_meter                  = [8.05, 8.7, 9.17, 9.9]  #8.05
frames_second                 = 25.0
update_speed                  = 1
roi_speed                     = [[219, 120], [228, 49], [59, 40], [11, 98]]
lanes                         = [
                                 [[284, 62], [0, 45], [0, 37], [279, 53]],
                                 [[290, 79],  [0, 58], [0, 47], [286, 66]],
                                 [[293, 100], [0, 76], [0, 62], [289, 83]],
                                 [[303, 121], [0, 95], [0, 80], [301, 102]]
                                 ]
max_speed                     = 80
high_density                  = 5

#Visualization
margin = 150

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
