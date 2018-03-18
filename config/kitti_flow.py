# Dataset
dataset_name                 = 'kitti'                          # Dataset name
dataset_path                 = 'datasets/kitti/image_0'         # Dataset path
gt_path                      = 'datasets/kitti/flow_noc'        # Ground truth path
results_path                 = 'datasets/kitti/results'

# Input Images
image_sequence              = '000045'              # '000045' or '000157']
image_type                   = 'png'                        # Input image type

# Block Matching Optical Flow
compensation                = 'backward'  # 'forward' instead
block_size                  = 8
search_area                 = 32          # Search area must be bigger than block size
dfd_norm_type               = 'l1'         # One of: 'l1', 'l2'

# Evaluate Optical Flow
evaluate                     = True
plot_optical_flow            = True
plot_prediction              = True
optical_flow_downsample      = 16

# Save results
save_results                 = True
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk