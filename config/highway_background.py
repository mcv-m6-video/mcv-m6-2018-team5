# Dataset
dataset_name                 = 'highway'                        # Dataset name
dataset_path                 = 'datasets/highway/input'         # Dataset path
gt_path                      = 'datasets/highway/groundtruth'   # Ground truth path
results_path                 = 'datasets/highway/results'

# Input Images
nr_images                    = 300
first_image                  = '001050'   # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Segmentation Metrics
segmentation_metrics          = True        # Precision, Recall, F1-Score
temporal_metrics              = True        # TP vs time, F1-score vs time
desynchronization             = True        # Apply desynchronization
desynchronization_frames      = [0, 5, 10]  # Nr frames to desynchronize

# Background Modelling
alpha                         = 1.1
rho                           = 0.5
modelling_method              = 'lsbp'  # {gaussian, adaptive, mog, mog2, gmg, lsbp}
evaluate_foreground           = False
evaluate_alpha_range          = [0.01, 5]
evaluate_alpha_values         = 100         # number of alpha values to evaluate
evaluate_rho_range            = [0.01, 5]   # range of rho values
evaluate_rho_values           = 100         # number of rho values to evaluate

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
