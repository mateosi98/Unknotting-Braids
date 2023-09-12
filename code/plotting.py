import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib.cm as cm  

root_dir = ""


tag_to_extract = "rollout/ep_len_mean"

all_steps = []
all_values = []
colors = cm.rainbow  

log_directories = []

for root, dirs, files in os.walk(root_dir):
    if any(file.startswith("events.out.tfevents.") for file in files):
        log_directories.append(root)

for i, log_directory in enumerate(log_directories):
    steps_list = []
    values_list = []

    for root, dirs, files in os.walk(log_directory):
        for file in files:
            if file.startswith("events.out.tfevents."):
                log_file_path = os.path.join(root, file)

                event_acc = event_accumulator.EventAccumulator(log_file_path)
                event_acc.Reload()

                if tag_to_extract in event_acc.Tags()["scalars"]:
                    tag_data = event_acc.Scalars(tag_to_extract)
                    steps = [event.step for event in tag_data]
                    values = [event.value for event in tag_data]

                    steps_list.extend(steps)
                    values_list.extend(values)

    color = colors(i / len(log_directories))

    plt.plot(steps_list, values_list, label=f"Model {i+1}", color=color)

    plt.xlim(0, 3_000_000)

plt.legend()

plt.xlabel("Steps")
plt.ylabel(tag_to_extract)
plt.yscale('log')
plt.title(f"Visualization of {tag_to_extract} Across Multiple Models")

plt.show()