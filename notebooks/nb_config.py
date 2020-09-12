if __name__ == '__main__' and __package__ is None:
    from os import sys, path

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    repo_path = path.dirname(path.dirname(path.dirname(path.abspath("__file__"))))
    sys.path.append(repo_path)

    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.precision", 6)
    pd.set_option('max_info_columns', 10)

    sns.set_context("talk")
    plt.style.use('ggplot')

    plt.rcParams['figure.figsize'] = (16, 8)
    plt.rcParams['figure.titlesize'] = 24

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['font.size'] = 12
    #plt.rcParams['legend.title_fontsize '] = 14
    plt.rcParams['legend.fontsize'] = 12



