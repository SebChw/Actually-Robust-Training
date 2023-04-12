from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import gc
import matplotlib

matplotlib.use("Agg")


class SourceSepPlotter:
    def __init__(self, saving_interval=10, y_axis_plot_limit=1.0):
        self.COLORS = [
            "#e6194b",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#bcf60c",
            "#fabebe",
        ]
        self.TEMP_PLOTS_PATH = Path("temp_plots")
        self.TEMP_PLOTS_PATH.mkdir(exist_ok=True)
        self.saving_interval = saving_interval
        self.y_axis_plot_limit = y_axis_plot_limit

    def update(self, lit_module):
        for stage_name, stage_losses in lit_module.song_losses.items():
            for song, instrument_losses in stage_losses.items():
                plot_path = self.TEMP_PLOTS_PATH / f"{song}.pickle"

                if (
                    not plot_path.exists()
                    or lit_module.current_epoch % self.saving_interval == 0
                ):
                    fig, ax = plt.subplots(1, 4, figsize=(30, 10), num=1, clear=True)
                    fig.suptitle(song, fontsize=16)
                    for i, (instrument, loss) in enumerate(instrument_losses.items()):
                        ax[i].set_title(instrument)
                        ax[i].set_xlabel("number of window")
                        ax[i].set_ylabel("L1")
                        ax[i].set_ylim([0, self.y_axis_plot_limit])

                else:
                    with open(plot_path, "rb") as f:
                        fig, ax = pickle.load(f)

                for i, (instrument, loss) in enumerate(instrument_losses.items()):
                    loss = np.trim_zeros(loss, "b")
                    ax[i].plot(
                        loss,
                        color=self.COLORS[lit_module.current_epoch % 10],
                        marker="o",
                        label=f"Epoch{lit_module.current_epoch}",
                    )
                    ax[i].legend()

                if lit_module.current_epoch % self.saving_interval == (
                    self.saving_interval - 1
                ):
                    lit_module.logger.experiment[
                        f"L1_losses/{stage_name}/{song}/{lit_module.current_epoch}"
                    ].upload(fig)

                with open(plot_path, "wb") as f:
                    pickle.dump((fig, ax), f)

                fig.clf()
                plt.close()
                gc.collect()
