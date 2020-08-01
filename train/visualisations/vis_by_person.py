from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

##### from https://stackoverflow.com/a/4700674/5140684 [for matplotlib legend] #######
fontP = FontProperties()
fontP.set_size('small')
####################### ######################### ################################

PLOT_DIR = 'vis_outputs/per_person/'


def drop_column_by_idx(df, index=None):
    df = df.drop(df.columns[index], axis=1)
    return df

class VisualizeStatsPerPerson():
    def __init__(self, dataname, df):
        self.dataname = dataname
        self.df = df
        if self.dataname == 'dsads' or self.dataname == 'pamap':
            self.person_column = 407 if self.dataname == 'dsads' else 244
            self.class_column = 406 if self.dataname == 'dsads' else 243
            self.df_by_mean_of_classes = drop_column_by_idx(self.get_mean_by_columns([self.df.columns[self.class_column]]),
                                                            self.person_column-1)
            self.df_by_mean_of_persons_and_classes = self.get_mean_by_columns([self.df.columns[self.person_column],
                                                                               self.df.columns[self.class_column]])

    def get_mean_by_columns(self, list_of_columns):
        return self.df.groupby(list_of_columns).mean()

    def plot_variance_by_persons(self):
        cosine_similarities, pearsons, spearmans, kendalls = [], [], [], []
        for i, new_df in self.df_by_mean_of_persons_and_classes.groupby(level=0):
            for idx, row in new_df.iterrows():
                class_num = idx[1]
                x = list(row)
                y = list(self.df_by_mean_of_classes.loc[class_num])
                cosine_similarities.append(cosine(x,y))
                pearsons.append(pearsonr(x,y)[0])
                kendalls.append(kendalltau(x,y)[0])
                spearmans.append(spearmanr(x,y)[0])
            #### uncomment if only one similarity to use ######
            # new_df['cosine'] = list_of_similarities
            # new_df.reset_index(inplace=True)
            # new_df.rename(columns={244:'persons', 243: 'classes'}, inplace=True)
            # new_df.boxplot(by='classes', column=['cosine'], grid=False)
            # plt.show()
            #####################################################

        self.df_by_mean_of_persons_and_classes['cosine distance from mean'] = cosine_similarities
        self.df_by_mean_of_persons_and_classes['pearson'] = pearsons
        self.df_by_mean_of_persons_and_classes['spearman'] = spearmans
        self.df_by_mean_of_persons_and_classes['kendalltau'] = kendalls
        self.df_by_mean_of_persons_and_classes.reset_index(inplace=True)
        self.df_by_mean_of_persons_and_classes.rename(columns={self.df_by_mean_of_persons_and_classes.columns[0]:'persons',
                                                               self.df_by_mean_of_persons_and_classes.columns[1]:'classes'},
                                                      inplace=True)
        myfig = plt.figure()
        fig = self.df_by_mean_of_persons_and_classes.boxplot(by='persons', column=['cosine distance from mean',
                                                                                   'pearson', 'spearman', 'kendalltau'],
                                                             grid=False)
        plt.title("box plot by correlation")
        plt.savefig(PLOT_DIR + f"{self.dataname}_boxplot.png")
        plt.close(myfig)

    def visualise_imbalance_by_persons(self, persons_list=None):
        '''
        Called for PAMAP/HAPT.
        :param df: dataframe read from mat files
        :param persons_list: list of person IDs, supplied for hapt dataset
        :return: None
        '''
        df = self.df.copy()
        if self.dataname in ['pamap', 'dsads', 'opp']:
            person_column = 244 if self.dataname == 'pamap' else 407 if self.dataname == 'dsads' else 461
            activity_column = 243 if self.dataname == 'pamap' else 406 if self.dataname == 'dsads' else 460
            df = df.rename(columns={df.columns[person_column]: 'Persons', df.columns[activity_column]: 'Activities'})
            df['Activities'] = df['Activities'].astype(int)
            df['Persons'] = df['Persons'].astype(int)
            ax = df.groupby([df.columns[person_column], df.columns[activity_column]]).size().unstack().plot(kind='bar',
                                                                                                            stacked=True,
                                                                                                            colormap='jet',
                                                                                                            )
            activity_labels = {1: 'sit', 2:'stand', 3:'lie on back', 4:'lie to right', 5:'ascend stairs', 6:'descend stairs',
           7: 'stand in elevator', 8: 'move in elevator', 9: 'walk in parking lot', 10:'walk flat at 4 km/h',
           11: 'walk inclined at 4 km/h', 12: 'run at 8 km/h', 13: 'stepper exercise', 14: 'cross trainer exercise',
           15: 'cycle horizontally', 16: 'cycle vertically', 17: 'row', 18: 'jump', 19: 'play basketball'}
            activity_names = list(activity_labels.values())
        elif self.dataname == 'hapt':
            df.insert(0, 'Persons', persons_list)
            ax = df.groupby(['Persons', 'AID']).size().unstack().plot(kind='bar',stacked=True, colormap='jet')
        else:
            ax = df.groupby(['AID']).size().plot(kind='bar',stacked=True, colormap='jet')
            # ax.set_xticklabels(['cook', 'eat', 'enter/leave house', 'living room activity', 'use toilet', 'use mirror',
            #                     'read', 'sleep', 'work'], rotation=50, fontsize=11, horizontalalignment='right')
        next_width = 0
        # for i, p in enumerate(ax.patches):
        #     width, height = p.get_width(), p.get_height()
        #     x, y = p.get_xy()
        #     if height > 0:
        #         ax.text(x + width / 2,
        #                 y + height / 2,
        #                 '{:.0f}'.format(height),
        #                 horizontalalignment='center',
        #                 verticalalignment='center')
        # L = plt.legend()
        # for i in range(0, len(activity_names)):
        #     L.get_texts()[i].set_text(activity_names[i])
        if self.dataname in ['pamap', 'dsads', 'hapt']:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(bbox_to_anchor=(1, 0.5), fontsize=13, title='Activity ID', loc='center left',
                      fancybox=True, shadow=True)
        import numpy as np
        ax.set_yticks(np.arange(0, 200, 30))
        plt.grid(True, linestyle=':', linewidth=0.3)
        plt.tick_params(axis='both', which='major', labelsize=13)
        # ax.yaxis.grid(True, which='minor', linestyle='-.', linewidth=0.25)
        ax.set_xlabel("Activity ID", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)

        ax.figure.savefig(PLOT_DIR + 'imbalance_vis_' + self.dataname + '.pdf', dpi=600,bbox_inches='tight')
        plt.show()