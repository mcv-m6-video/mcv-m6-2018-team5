# Dataset
dataset_name                 = 'traffic'                        # Dataset name
dataset_path                 = 'datasets/traffic/input'         # Dataset path
gt_path                      = 'datasets/traffic/groundtruth'   # Ground truth path
results_path                 = 'datasets/traffic/results/stabilization'

# Input Images
nr_images                    = 100
first_image                  = '000950'     # Fist image filename
image_type                   = 'jpg'        # Input image type
gt_image_type                = 'png'        # Ground truth image type
result_image_type            = 'png'

# Block Matching Optical Flow
compensation                = 'forward'  # 'forward' instead
block_size                  = 4
search_area                 = 4          # Search area must be bigger than block size
dfd_norm_type               = 'l1'        # One of: 'l1', 'l2'
optimize_block_matching     = False        # Whether to optimize parameters of block matching or not
block_size_range            = [4, 8, 16, 32, 64]
search_area_range           = [4, 8, 16, 32, 64]
sota_opt_flow               = True
sota_opt_flow_option        = 'opencv'

sota_video_stab             = False

# Save results
save_results                 = True        # Save Log file
output_folder                = 'results'   # Output folder to save the results of the test
save_plots                   = True        # Save the plots to disk
