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

modelling_method              = 'adaptive'      # adaptive, non-adaptive
color_images                  = True            # Use RGB, HSV color channels
color_space                   = "RGB"           # RGB, HSV
evaluate_foreground           = True
evaluate_alpha_range          = [0, 25]     # range of alpha values
evaluate_alpha_values         = 50          # number of alpha values to evaluate
evaluate_rho_range            = [0, 1]      # range of rho values
evaluate_rho_values           = 20          # number of rho values to evaluate
find_best_parameters          = False
plot_back_model               = False

# Foreground Modelling
four_connectivity             = False
AUC_area_filtering            = True		 # Compute AUC vs P pixels for this sequence
P_pixels_range                = [0, 1000]    # range of P pixels
P_pixels_values               = 40

task_name                     = 'task3'      # else task1, task2
opening_strel                 = 'rectangle'
opening_strel_size            = 3
closing_strel                 = 'rectangle'
closing_strel_size            = 5

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
