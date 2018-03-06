# Dataset
dataset_name                 = 'highway'                        # Dataset name
dataset_path                 = 'datasets/highway/input'         # Dataset path
gt_path                      = 'datasets/highway/groundtruth'   # Ground truth path
results_path                 = 'datasets/highway/results/color_adaptive'

# Input Images
nr_images                    = 300
first_image                  = '001050'   # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Background Modelling
alpha                         = 6
rho                           = 0.21

modelling_method              = 'non-adaptive'  # adaptive instead
color_images                  = True        # Use RGB, HSV color channels
color_space                   = "HSV"       # RGB, HSV
evaluate_foreground           = True
evaluate_alpha_range          = [0, 6]      # range of alpha values
evaluate_alpha_values         = 60          # number of alpha values to evaluate
evaluate_rho_range            = [0, 1]      # range of rho values
evaluate_rho_values           = 20          # number of rho values to evaluate
find_best_parameters          = True
plot_back_model               = False

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk

