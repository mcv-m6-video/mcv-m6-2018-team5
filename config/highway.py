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

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
