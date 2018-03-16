# Dataset
dataset_name                 = 'kitti'                          # Dataset name
dataset_path                 = 'datasets/kitti/image_0'         # Dataset path
gt_path                      = 'datasets/kitti/flow_noc'        # Ground truth path
results_path                 = 'datasets/kitti/results'

# Input Images
image_sequences              = ['000045_10', '000157_10']   # List of the sequences
image_type                   = 'png'                        # Input image type

# Evaluate Optical Flow
evaluate                     = False
plot_optical_flow            = False
optical_flow_downsample      = 16

# Save results
save_results                 = True
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
