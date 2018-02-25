# Dataset
dataset_name                 = 'kitti'                          # Dataset name
dataset_path                 = 'datasets/kitti/image_0'         # Dataset path
gt_path                      = 'datasets/kitti/flow_noc'        # Ground truth path
results_path                 = 'datasets/kitti/results'

# Input Images
nr_images                    = 2
first_image                  = '000047_10'   # Fist image filename
image_type                   = 'png'         # Input image type
gt_image_type                = 'png'         # Ground truth image type
result_image_type            = 'png'

# Compute Metrics
compute_metrics              = False
# Compute Optical flow
optical_flow                 = True

# Save results
save_results                 = True
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
