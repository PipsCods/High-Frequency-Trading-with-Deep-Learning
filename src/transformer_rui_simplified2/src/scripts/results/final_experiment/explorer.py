import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_figures(experiments: list[str]):
    folder = "losses_figures/"
    os.makedirs(folder, exist_ok=True)

    for experiment in experiments:
        data = pd.read_csv(experiment)

        if 'losses' in experiment:
            epochs = range(len(data))

            plt.figure(figsize=(10, 8))
            plt.plot(epochs, data['train'], label="Train Loss")
            plt.plot(epochs, data['test'], label="Test Loss")
            plt.plot(epochs, data['regime_acc'], label="Directional Accuracy")
            plt.plot(epochs, data['return_loss'], label="Loss on returns reg.")
            plt.plot(epochs, data['regime_loss'], label="Loss on regime class.")
            plt.title(experiment)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(folder, f"{experiment}.png"))
            plt.close()

        else:
            pass

if __name__ == '__main__':
    folder_path = r"."
    experiments = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    experiments = [xp for xp in experiments if 'losses' in xp]
    generate_figures(experiments)