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

# Segmentation Metrics
segmentation_metrics          = True        # Precision, Recall, F1-Score
temporal_metrics              = True        # TP vs time, F1-score vs time
desynchronization             = True        # Apply desynchronization
desynchronization_frames      = [0, 5, 10]  # Nr frames to desynchronize

#Background Modelling
alpha                         = 1.1
rho                           = 0.5
modelling_method              = 'gaussian'  # adaptive instead
evaluate_foreground           = True
evaluate_alpha_range          = [0.01, 5]

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk