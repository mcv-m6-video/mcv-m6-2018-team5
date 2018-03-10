# Dataset
dataset_name                 = 'fall'                        # Dataset name
dataset_path                 = 'datasets/fall/input'         # Dataset path
gt_path                      = 'datasets/fall/groundtruth'   # Ground truth path
results_path                 = 'datasets/fall/results/'

# Input Images
nr_images                    = 100
first_image                  = '001460'     # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Background Modelling
alpha                         = 3.533
rho                           = 0.0526

modelling_method              = 'non-adaptive'  # gaussian, adaptive
color_images                  = False        # Use RGB, HSV color channels
color_space                   = "RGB"       # RGB, HSV
evaluate_foreground           = True
evaluate_alpha_range          = [0, 65]      # range of alpha values
evaluate_alpha_values         = 150          # number of alpha values to evaluate
evaluate_rho_range            = [0, 1]      # range of rho values
evaluate_rho_values           = 20          # number of rho values to evaluate
find_best_parameters          = False
plot_back_model               = False

# Foreground Modelling
connectivity                  = 4
hole_size_range               = [0, 100]    # range of hole size values

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
