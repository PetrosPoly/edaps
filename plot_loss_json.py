import json
import pandas as pd
import matplotlib.pyplot as plt

data =[]
# Replace 'your_file.json' with the path to your JSON file
with open('edaps_experiments/exp-00001/work_dirs/local-exp00001/231129_1502_syn2cs_dacs_rcs001_cpl_maskrcnn_mitb5_poly10warm_s0_f3247/20231129_150249.log_me.json', 'r') as file:
   for line in file:
        data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure 'iter' and 'decode.loss_seg' columns exist
if 'iter' in df.columns and 'decode.loss_seg' in df.columns:
    # Plotting
    df.plot(x='iter', y='decode.loss_seg', kind='line')
    plt.xlabel('Iteration')
    plt.ylabel('Decode Loss (Segmentation)')
    plt.title('Decode Loss Segmentation per Iteration')
    plt.show()
