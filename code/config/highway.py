# Dataset
dataset_name                 = 'highway'                        # Dataset name
dataset_path                 = 'datasets/highway/input'         # Dataset path
gt_path                      = 'datasets/highway/groundtruth'   # Groung truth path
results_path                 = 'datasets/highway/results_testAB_changedetection'

# Input Images
nr_images                    = 200
first_image                  = '001201'   # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Compute Metrics
compute_metrics              = True
# Compute Optical flow
optical_flow                 = False

# Save results
save_results                 = True
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
