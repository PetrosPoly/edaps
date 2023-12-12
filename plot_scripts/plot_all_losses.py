import os
import subprocess
import os
import glob

# Use glob to find all JSON files in the folder

# /media/suman/CVLHDD/apps/experiments/mmgeneration_experiments_on_euler_backup/euler-exp00004/231016_2316_lamp_pix2pix_dnc_3_dbc_32_glr_0.0002_dlr_2e-06_9cc2a/
src_root ='losses/losses_experiments_8_loss_01/exp-00001/work_dirs'
plot_template_file_name = 'plot_train_losses/plot_template.sh'
# src_root = '/media/suman/CVLHDD/apps/experiments/mmgeneration_experiments_on_euler_backup'
# plot_template_file_name = 'plot_train_losses/plot_template.sh'
# plot_template_file_name = '/home/suman/code/mmgeneration/plot_train_losses/plot_template.sh'

exp_name = 'local-exp00001'
exp_path1 = os.path.join(src_root, exp_name)
exp_list = os.listdir(exp_path1)
for sub_exp_name in exp_list:
    print(sub_exp_name)
    json_path = os.path.join(exp_path1, sub_exp_name)
    json_files_path = glob.glob(os.path.join(json_path, '*.log.json'))
    json_file_path = json_files_path[0]
    out_file_loss_disc = f'{json_path}/loss_disc.pdf'
    out_file_loss_gen = f'{json_path}/loss_gen.pdf'
    out_file_loss_pix = f'{json_path}/loss_pix.pdf'
    out_file_loss_total = f'{json_path}/loss_total.pdf'
    # Generate submission script
    with open(plot_template_file_name, 'r') as f:
        submit_template_str = f.read()
        # exec_cmd = get_exec_cmd(cfg)
        submit_str = submit_template_str.format(
            json_file_path=json_file_path,
            out_file_loss_disc=out_file_loss_disc,
            out_file_loss_gen=out_file_loss_gen,
            out_file_loss_pix=out_file_loss_pix,
            out_file_loss_total=out_file_loss_total
        )
    submit_file = f'plot_train_losses/{exp_name}_{sub_exp_name}_plot.sh'
    # create the job shell file
    with open(submit_file, 'w') as f:
        f.write(submit_str)
    try:
        # Run the Bash script
        subprocess.run(['bash', submit_file], check=True)
        print(f"Successfully ran '{submit_file}'")
    except subprocess.CalledProcessError as e:
        print(f"Error running '{submit_file}': {e}")
    except FileNotFoundError:
        print(f"'{submit_file}' not found. Make sure the path is correct.")
