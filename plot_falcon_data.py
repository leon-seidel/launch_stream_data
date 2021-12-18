import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt


def get_data(csv_stage):
    df = pd.read_csv(csv_stage)

    df = df.dropna()

    # df = df[(np.abs(stats.zscore(df)) < 1.8).all(axis=1)]     # Delete outliers in all columns
    df = df[(np.abs(stats.zscore(df.v)) < 1.8)]                 # Delete velocity outliers
    df = df[(np.abs(stats.zscore(df.h)) < 1.8)]                 # Delete altitude outliers

    return df


def show_stuff(csv_name, plot_type):

    for i in range(0, 2):
        for name in csv_name:
            df = get_data(name)

            if plot_type == "plot":
                if i == 0:
                    plt.plot(df.t, df.v)
                elif i == 1:
                    plt.plot(df.t, df.h)
            elif plot_type == "scatter":
                if i == 0:
                    plt.scatter(df.t, df.v)
                elif i == 1:
                    plt.scatter(df.t, df.h)
        plt.grid()
        plt.show()


csv_names = ["falcon_data_stage1.csv"]
plot_or_scatter = "scatter"

show_stuff(csv_names, plot_or_scatter)
