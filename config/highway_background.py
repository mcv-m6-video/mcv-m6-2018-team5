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
color_images                  = True            # Use RGB, HSV color channels
color_space                   = "RGB"           # RGB, HSV
evaluate_foreground           = True
evaluate_alpha_range          = [0, 10]     # range of alpha values
evaluate_alpha_values         = 50          # number of alpha values to evaluate
evaluate_rho_range            = [0, 1]      # range of rho values
evaluate_rho_values           = 20          # number of rho values to evaluate
find_best_parameters          = False
plot_back_model               = False

# Foreground Modelling
four_connectivity             = False
AUC_area_filtering            = True		 # Plot AUC vs P pixels
P_pixels_range                = [0, 1500]    # range of P pixels
P_pixels_values               = 30

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk

