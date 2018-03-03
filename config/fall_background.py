# Dataset
dataset_name                 = 'fall'                        # Dataset name
dataset_path                 = 'datasets/fall/input'         # Dataset path
gt_path                      = 'datasets/fall/groundtruth'   # Ground truth path
results_path                 = 'datasets/fall/results'

# Input Images
nr_images                    = 100
first_image                  = '001460'     # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Background Modelling
alpha                         = 1
rho                           = 0.5
modelling_method              = 'gaussian'  # adaptive instead
evaluate_foreground           = False
evaluate_alpha_range          = [0.01, 5]   # range of alpha values
evaluate_alpha_values         = 100         # number of alpha values to evaluate
evaluate_rho_range            = [0.01, 1]   # range of rho values
evaluate_rho_values           = 100         # number of rho values to evaluate
find_best_parameters          = True

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
