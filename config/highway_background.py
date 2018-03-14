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
evaluate_alpha_range          = [1, 25]     # range of alpha values
evaluate_alpha_values         = 100         # number of alpha values to evaluate
evaluate_rho_range            = [0, 1]      # range of rho values
evaluate_rho_values           = 20          # number of rho values to evaluate
find_best_parameters          = False
plot_back_model               = False

# Foreground Modelling
four_connectivity             = False
AUC_area_filtering            = False		 # Plot AUC vs P pixels
P_pixels_range                = [0, 1000]    # range of P pixels
P_pixels_values               = 40

task_name                     = 'task3'      # else task1, task2
opening_strel                 = 'square'
opening_strel_size            = 5
closing_strel                 = 'diamond'
closing_strel_size            = 3
area_filtering                = True
area_filtering_P              = 128
shadow_remove                 = True

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
