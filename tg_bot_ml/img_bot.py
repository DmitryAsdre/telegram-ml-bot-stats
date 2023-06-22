from collections import defaultdict

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .tg_bot import TGSummaryWriterBase

class TGImgSummaryWriter(TGSummaryWriterBase):
    def __init__(self,  path_to_credentials, experiment_name):
        super().__init__(path_to_credentials, experiment_name)

        self.metrics = {}        
        self.images = defaultdict(lambda : list())
        self.description = None

    def add_image(self, np_img, group=None):
        self.images[group].append(np_img)
    
    def add_description(self, description):
        self.description = description

    def add_scalar(self, name, value, step, group=None):
            if (name, group) in self.metrics:
                self.metrics[(name, group)][0].append(value)
                self.metrics[(name, group)][1].append(step)
            else:
                self.metrics[(name, group)] = [[value], [step]]

    def collect_statistics(self):
        metrics_statistics = []
        for name, group in self.metrics:
            metrics_data = self.metrics[(name, group)]

            cur_metric_stats = {"metric" : name}
            cur_metric_stats['mean'] = np.mean(metrics_data[0])
            cur_metric_stats['max'] = np.max(metrics_data[0])
            cur_metric_stats['min'] = np.min(metrics_data[0])
            cur_metric_stats['argmax'] = metrics_data[1][np.argmax(metrics_data[0])]
            cur_metric_stats['argmin'] = metrics_data[1][np.argmin(metrics_data[0])]
            cur_metric_stats['mean -5:'] = np.mean(metrics_data[0][-5:])

            metrics_statistics.append(cur_metric_stats)

        return pd.DataFrame.from_records(metrics_statistics)
    
    def collect_images(self, figsize=(10, 8)):
        stats_imgs = defaultdict(lambda : list())
        
        for name, group in self.metrics:
            metrics_data = self.metrics[(name, group)]
            fig, ax = plt.subplots()
            ax.set_title(f'Record name : {name}')
            ax.set_xlabel('step')
            ax.set_ylabel(name)
            ax.plot(metrics_data[1], metrics_data[0], 'g', label=name)

            ax.legend()

            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            stats_imgs[group].append(data)
            plt.close(fig)
        
        return stats_imgs
    
    def send(self, send_statistics=True, send_images=True, send_plots=True):
        self.send_delimiter()
        if send_statistics:
            statistics = self.collect_statistics()
            self.send_pandas_df(statistics)
        if send_plots:
            plots = self.collect_images()
            for group_name in plots:
                group_plots = plots[group_name]
                self.send_numpy_images(group_plots)
        if send_images:
            for group_name in self.images:
                group_images = self.images[group_name]
                self.send_numpy_images(group_images)

        self.images = defaultdict(lambda : list())
    





        
    

