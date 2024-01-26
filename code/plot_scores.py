import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 




def line_plot(folder, area_keys, border_keys, last_filters, n_Ks, save=False):
    for area_key, border_key in zip(area_keys, border_keys):
        mult = 100/np.array(area_score["tot_points"]) if area_key != "area_scores" else 100
        fig, axs = plt.subplots(1,2 , figsize=[15,5])
        for last_filter in last_filters:
            area_scores = []
            border_scores = []
            for n_K in n_Ks:
                area_score = np.load(folder + f"filter{last_filter}/clustering/scores/area_scores_cluster_dnb_l95_z50_ps128_band29_filter{last_filter}_K{n_K}_res32_thr0.npy", allow_pickle=True).item()
                border_score = np.load(folder + f"filter{last_filter}/clustering/scores/border_scores_cluster_dnb_l95_z50_ps128_band29_filter{last_filter}_K{n_K}_res32_thr0.npy", allow_pickle=True).item()
                area_scores.append(np.mean(np.array(area_score[area_key])*mult))
                border_scores.append(np.mean(np.array(border_score[border_key])*mult))
                        
            axs[0].plot(n_Ks, area_scores, label=f"Last filter: {last_filter}")
            axs[0].set_title("Area")
            axs[1].plot(n_Ks, border_scores, label=f"Last filter: {last_filter}")
            axs[1].set_title("Border")
            axs[0].set_ylabel("Accuracy [%]")
            axs[0].set_xlabel(f"Kmeans $n_K$")


        fig.suptitle(' '.join(area_key.split("_")[1:]), fontsize=30)


        axs[0].legend()
        axs[1].legend()

        if save:
            plt.savefig(f"/uio/hume/student-u37/fslippe/master_project/figures/score_plots/line_plot_{area_key}.jpg", dpi=200)
        plt.show()


def bar_plot(folder, area_keys, border_keys, last_filters, n_Ks, save=False):
    bar_width = 0.3  # adjust as needed
    index = np.arange(len(n_Ks))
    for area_key, border_key in zip(area_keys, border_keys):
        mult = 100/np.array(area_score["tot_points"]) if area_key != "area_scores" else 100

        fig, axes = plt.subplots(2,1)
        for i, last_filter in enumerate(last_filters):
            area_scores = []
            border_scores = []
            for n_K in n_Ks:
                area_score = np.load(f"{folder}filter{last_filter}/clustering/scores/area_scores_cluster_dnb_l95_z50_ps128_band29_filter{last_filter}_K{n_K}_res32_thr0.npy", allow_pickle=True).item()
                border_score = np.load(f"{folder}filter{last_filter}/clustering/scores/border_scores_cluster_dnb_l95_z50_ps128_band29_filter{last_filter}_K{n_K}_res32_thr0.npy", allow_pickle=True).item()
                area_scores.append(np.mean(np.array(area_score[area_key])*mult))
                border_scores.append(np.mean(np.array(border_score[border_key])*mult))

            # Plot bar graph with an offset for each filter and with a label
            axes[0].bar(index + i*bar_width, area_scores, bar_width, label=f"Last filter: {last_filter}")
            axes[1].bar(index + i*bar_width, border_scores, bar_width, label=f"Last filter: {last_filter}")

        # Set labels, title and xticks
        for ax in axes:
            ax.set_xlabel("Kmeans $n_K$")
            ax.set_ylabel("Accuracy [%]")
            ax.set_xticks(index + bar_width * (len(last_filters) - 1) / 2)  # Position of xticks
            ax.set_xticklabels(n_Ks)

        axes[0].set_title("Area")
        axes[1].set_title("Border")
        axes[0].legend(loc="lower right")
        axes[1].legend(loc="lower right")
        fig.suptitle(' '.join(area_key.split("_")[1:]), fontsize=20)


        fig.tight_layout()
        if save:
            plt.savefig(f"/uio/hume/student-u37/fslippe/master_project/figures/score_plots/bar_plot_{area_key}.jpg", dpi=200)
        plt.show()

def heatmap_plot(folder, area_keys, border_keys, last_filters, n_Ks, save=False):
    for area_key, border_key in zip(area_keys, border_keys):
        area_scores_matrix = np.zeros((len(last_filters), len(n_Ks)))
        border_scores_matrix = np.zeros((len(last_filters), len(n_Ks)))
        mult = 100/np.array(area_score["tot_points"]) if area_key != "area_scores" else 100
        for i, last_filter in enumerate(last_filters):
            for j, n_K in enumerate(n_Ks):
                area_score = np.load(f"{folder}filter{last_filter}/clustering/scores/area_scores_cluster_dnb_l95_z50_ps128_band29_filter{last_filter}_K{n_K}_res32_thr0.npy",
                                    allow_pickle=True).item()
                border_score = np.load(f"{folder}filter{last_filter}/clustering/scores/border_scores_cluster_dnb_l95_z50_ps128_band29_filter{last_filter}_K{n_K}_res32_thr0.npy", 
                                    allow_pickle=True).item()
                area_scores_matrix[i][j] = np.mean(np.array(area_score[area_key])*mult)
                border_scores_matrix[i][j] = np.mean(np.array(border_score[border_key])*mult)

        # Now plot the heatmaps.
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        sns.heatmap(area_scores_matrix, ax=axes[0], annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=n_Ks, yticklabels=last_filters, cbar_kws={'label': 'Accuracy [%]'})

        axes[0].set_title("Area Heatmap")
        axes[0].set_xlabel("$n_K$")
        axes[0].set_ylabel("Last Filter")

        sns.heatmap(border_scores_matrix, ax=axes[1], annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=n_Ks, yticklabels=last_filters, cbar_kws={'label': 'Accuracy [%]'})
        axes[1].set_title("Border Heatmap")
        axes[1].set_xlabel("$n_K$")
        axes[1].set_ylabel("Last Filter")
        fig.suptitle(' '.join(area_key.split("_")[1:]), fontsize=30)


        fig.tight_layout()  # Adjust layout to minimize overlaps.
        if save:
            plt.savefig(f"/uio/hume/student-u37/fslippe/master_project/figures/score_plots/heatmap_plot_{area_key}.jpg", dpi=200)
        plt.show()


def main():
    folder = "/uio/hume/student-u37/fslippe/data/models/patch_size128/"
    last_filters = [32, 64, 128] 
    n_Ks = [10, 11, 12, 13, 14, 15, 16]
    area_keys = ["area_scores",'area_true_positive_scores', 'area_false_positive_scores', 'area_true_negative_scores', 'area_false_negative_scores', 'area_true_prediction_scores', 'area_false_prediction_scores']
    border_keys = ["border_scores", 'border_true_positive_scores', 'border_false_positive_scores', 'border_true_negative_scores', 'border_false_negative_scores', 'border_true_prediction_scores', 'border_false_prediction_scores']
    #bar_plot(folder, area_keys, border_keys, last_filters, n_Ks, save=True)
    heatmap_plot(folder, area_keys, border_keys, last_filters, n_Ks, save=True)

if __name__ == "__main__":
    main()