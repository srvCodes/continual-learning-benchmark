from itertools import chain

# from tsnecuda import TSNE
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

fontP = FontProperties()
fontP.set_size('small')

OUT_DIR = 'vis_outputs/exemp_vis/'
PAMAP_COLOR_DICT = dict({1: 'black', 2: 'red', 3: 'gold', 4: 'deepskyblue', 5: 'grey',
                         6: 'olive', 7: 'indigo', 12: 'deeppink', 13: 'orange',
                         16: 'lightblue', 17: 'teal', 24: 'brown'})
DSADS_COLOR_DICT = dict({1: 'black', 2: 'red', 3: 'gold', 4: 'deepskyblue', 5: 'grey',
                         6: 'olive', 7: 'indigo', 8: 'deeppink', 9: 'orange',
                         10: 'lightblue', 11: 'teal', 12: 'brown', 13: 'lime', 14: 'mediumblue',
                         15: 'mediumspringgreen', 16: 'lightsalmon', 17: 'lightsteelblue',
                         18: 'orchid', 19: 'sandybrown'})


def scatter_plot_exemps(label_to_features_all, label_to_indices_exemp, virtual_map, original_map, strategy, data_name):
    all_values = np.array(list(chain(*label_to_features_all.values())))
    pca_50 = PCA(n_components=50)
    all_values = pca_50.fit_transform(all_values)
    label_to_indices_adjusted = {label: np.array(indices) + len(label_to_indices_exemp[idx - 1]) if idx > 0 else indices
                                 for idx, (label, indices) in enumerate(label_to_indices_exemp.items())}
    color_coding = PAMAP_COLOR_DICT if data_name == 'pamap' or data_name == 'hapt' else DSADS_COLOR_DICT
    tsne_feats = TSNE(n_components=2, perplexity=15, learning_rate=100).fit_transform(all_values)
    sns_plot = sns.scatterplot(tsne_feats[:, 0], tsne_feats[:, 1], color='grey')
    for label, indices in label_to_indices_adjusted.items():
        label = original_map[virtual_map[label]]
        sns.scatterplot(tsne_feats[indices, 0], tsne_feats[indices, 1], color=color_coding[label], legend='full')

    fig = sns_plot.get_figure()
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position
    sns_plot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                    prop=fontP)
    fig.savefig(OUT_DIR + f'{data_name}_{strategy}_exemps_{len(label_to_features_all)}.png')
    plt.show()
    fig.clf()
