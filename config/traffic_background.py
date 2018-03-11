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

modelling_method              = 'adaptive'  # adaptive instead
color_images                  = True            # Use RGB, HSV color channels
color_space                   = "RGB"           # RGB, HSV
evaluate_foreground           = True
evaluate_alpha_range          = [0, 25]         # range of alpha values
evaluate_alpha_values         = 150             # number of alpha values to evaluate
evaluate_rho_range            = [0, 1]          # range of rho values
evaluate_rho_values           = 20              # number of rho values to evaluate
find_best_parameters          = False
plot_back_model               = False

# Foreground Modelling
four_connectivity             = False
hole_size_range               = [0, 100]    # range of hole size values

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk

