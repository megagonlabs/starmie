from matplotlib import *
from matplotlib import pyplot as plt
import numpy as np


fsize = 20
tsize = 14
# =============================================================================
# Plotting of Figures for the paper
# =============================================================================  
def plotMapFig(benchmark, map_dict):
    ''' Plot the MAP scores as bar chart (NOT USED)
    Args:
        benchmark (str)
        map_dict (dict): stores each method with their associated map scores
    '''
    labels = {"d3l":r"$D^{3}L$",
        "SANTOS": r"SANTOS",
        "Starmie": r"Starmie",
        "SingleCol": r"SingleCol", 
        "SATO":r"SATO",
        "Sherlock": r"Sherlock"
        }
    
    # ========== MAP ==========
    x = []
    for method in map_dict.keys():
        x.append(labels[method])
    y = list(map_dict.values())
    
    x_ticks = np.arange(len(x))
    width = 0.7
    fig, ax = plt.subplots()
    ax.set_ylabel('MAP@k', fontsize=tsize)
    ax.set_xlabel('Method', fontsize=tsize)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x)
    if benchmark in ['santos', 'tus_large']:
        plt.ylim(0.4, 1.03)
    elif benchmark == 'tus_small':
        plt.ylim(0.75, 1.01)
    # Annotate each bar with their MAP score
    pps = ax.bar(x_ticks, y, width, label='MAP@k')
    for p in pps:
        height = p.get_height()
        ax.annotate('{}'.format(height),
            xy=(p.get_x() + p.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom')
    # save the figure to a local path
    fig.savefig('../../Starmie/%s_map.pdf' % (benchmark))
    plt.show()
    
    
    
def plotJointFig(k, benchmark, precision_list_dict, recall_list_dict, ideal_list):
    ''' Plot P@K and R@K figures for a specified benchmark
        Saves plot to a local filepath, and shows the path (the legend is hidden)
    Args:
        k (list): list of k values associated with each score
        benchmark (str): e.g. 'santos', 'tus_small'
        precision_list_dict (dict): With each method as key, its value is the list of precision scores for each k
        recall_list_dict (dict): With each method as key, its value is the list of recall scores for each k
        ideal_list: list of IDEAL recall scores for each k
    '''
    # number of methods to compare
    col_number = 6
    # Formatting / Styling choices
    colors = {"d3l":"#e52638",
                "SANTOS":"#777777",
                "Starmie": "royalblue",
                "SingleCol": "#68affc",
                "SATO": "#699f3c",
                "Sherlock": "darkgoldenrod"
                }
    linestyles = {"d3l":"dashed",
                "SANTOS":"dotted",
                "Starmie": "solid",
                "SingleCol": "dashdot",
                "SATO": (0, (3, 1, 1, 1)),
                "Sherlock": (0, (3, 1, 1, 1, 1, 1))
                }

    labels = {"d3l":r"$D^{3}L$",
            "SANTOS": r"SANTOS",
            "Starmie": r"Starmie",
            "SingleCol": r"SingleCol", 
            "SATO":r"SATO",
            "Sherlock": r"Sherlock"
            }

    markers = {"d3l":"^",
            "SANTOS":"o",
            "Starmie": "s",
            "SingleCol": "*",
            "SATO": "p",
            "Sherlock": "+"
            }
    # ========== PRECISION/RECALL with LEGEND ==========
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    for ax in axes:
        # formatting for both P@K and R@K graphs: x-axis is labeled with "k", add grids, set font sizes
        ax.set_xlabel("k", fontsize=fsize)
        ax.grid(linestyle = '--', linewidth = 0.5)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(tsize)
    plt.rc('legend', fontsize=tsize)
    
    # Plot the Precision@K graph
    axes[0].set_ylabel("P@k", fontsize=fsize)
    for which_method in precision_list_dict:
        axes[0].plot(k, precision_list_dict[which_method], color = colors[which_method], linestyle = linestyles[which_method], linewidth = 2, label = labels[which_method], marker=markers[which_method], markersize = 10)
    
    # Plot the Recall@K graph, along with IDEAL recall
    axes[1].set_ylabel("R@k", fontsize=fsize)
    for which_method in recall_list_dict:
        axes[1].plot(k, recall_list_dict[which_method], color = colors[which_method], linestyle = linestyles[which_method], linewidth = 2, label = labels[which_method], marker=markers[which_method], markersize = 10)
    axes[1].plot(k, ideal_list, color = "black", label = "IDEAL", linewidth = 3)
    
    # Add Legend
    handles, labels = axes[1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), ncol=col_number+1, loc='upper center')
    fig.tight_layout()

    # Save figure to a local path
    fig.savefig('../../Starmie/%s_P_R.pdf' % (benchmark), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    
    
def plotScalFig(k, dl_sizes, benchmark, scal_k, scal_size):
    ''' Plot the scalability figures for a specified benchmark
        Saves plot to a local filepath, and shows the path (the legend is hidden)
    Args:
        k (list): list of k values associated with each score
        dl_sizes (list): list of data lake sizes for x_axis of scalability graph for varying DL size
        benchmark (str): e.g. 'real', 'wdc'
        scal_k (dict): With each technique as key, its value is the query times (in ms)
        scal_size (dict): With each technique as key, its value is the query times (in ms)
    '''
    # number of methods to compare
    col_number = 4
    # Formatting / Styling choices
    colors = {"Linear":"royalblue",
                "Bounds":"green",
                "LSH": "red",
                "HNSW": "darkgoldenrod"
                }
    linestyles = {"LSH":"dashed",
                "Bounds":"dotted",
                "Linear": "solid",
                "HNSW": "dashdot"
                }

    labels = {"Linear":"Linear",
            "Bounds": "Bounds",
            "LSH": "LSH Index",
            "HNSW": "HNSW Index"
            }

    markers = {"LSH":"^",
            "Bounds":"o",
            "Linear": "s",
            "HNSW": "*"
            }
    
    x_axis_labels = ['K', 'Data Lake Size (# tables / # attributes)']
    # ========== SCALABILITY with LEGEND ==========
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    for ind, ax in enumerate(axes):
        # formatting for graphs: x-axis is labeled with "k", add grids, set font sizes
        ax.set_xlabel(x_axis_labels[ind], fontsize=fsize)
        ax.set_ylabel("Average Query Time (sec)", fontsize=fsize)
        ax.grid(linestyle = '--', linewidth = 0.5)    
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(tsize)
    plt.rc('legend', fontsize=tsize)
    
    # Plot the graph for varying k's as x-axis
    for which_method in scal_k:
        # if we want to plot in seconds. Otherwise, scal_k[which_method] is in ms
        scal_method = [int(qt)/1000 for qt in scal_k[which_method]]
        axes[0].plot(k, scal_method, color = colors[which_method], linestyle = linestyles[which_method], linewidth = 2, label = labels[which_method], marker=markers[which_method], markersize = 10)
    
    # Plot the graph for scalability, which growing data lake
    for which_method in scal_size:
        # if we want to plot in seconds. Otherwise, scal_k[which_method] is in ms
        scal_method = [int(qt)/1000 for qt in scal_size[which_method]]
        axes[1].plot(dl_sizes, scal_method, color = colors[which_method], linestyle = linestyles[which_method], linewidth = 2, label = labels[which_method], marker=markers[which_method], markersize = 10)

    # Add legend
    handles, labels = axes[1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, bbox_to_anchor=(0.5, 1.1), ncol=col_number+1, loc='upper center')
    fig.tight_layout()

    # Save plot to local file path
    fig.savefig('../../Starmie/%s_scal_sec.pdf' % (benchmark), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    '''
    Plot the experimental results figures, shown in the paper
    '''
    # ========== Metrics Dictionaries ==========
    # ==========================================
    ''' Plot the Performance metrics for each benchmark: SANTOS, TUS Small, TUS Large '''
    # ---------- 1. SANTOS Benchmark -----------
    precision_dict_santos = {'Starmie': [1, 1, 0.992, 0.991, 0.984], 'SingleCol': [1, 0.927, 0.896, 0.869, 0.798], 'SATO': [1, 0.913, 0.872, 0.846, 0.806], 'Sherlock': [1, 0.833, 0.772, 0.726, 0.672], 'SANTOS': [0.98, 0.947, 0.936, 0.926, 0.908], 'd3l': [0.5, 0.467, 0.512, 0.546, 0.576]}
    map_dict_santos = {'Starmie': 0.993, 'SingleCol': 0.891, 'SATO': 0.878, 'Sherlock': 0.782, 'SANTOS': 0.93, 'd3l': 0.523}
    recall_dict_santos = {'Starmie': [0.075,0.225,0.372,0.52,0.737], 'SingleCol': [0.075,0.208,0.333,0.451,0.588], 'SATO': [0.08,0.203,0.322,0.436,0.594], 'Sherlock': [0.08,0.185,0.284,0.373,0.493], 'SANTOS': [0.074,0.215,0.353,0.49,0.69], 'd3l': [0.037,0.099,0.185,0.278,0.422]}
    ideal_santos = [0.08,0.23,0.38,0.53,0.75]
    k_santos = [1,3,5,7,10]
    
    # plotJointFig(k_santos, 'santos', precision_dict_santos, recall_dict_santos, ideal_santos)
    # plotMapFig('santos', map_dict_santos)
    
    # ---------- 2. TUS Small Benchmark -----------
    precision_dict_tus_small = {'Starmie': [0.998,0.995,0.993,0.989,0.984,0.977], 'SingleCol': [0.977,0.97,0.956,0.944,0.927,0.907], 'SATO': [0.972,0.962,0.962,0.961,0.96,0.956], 'Sherlock': [0.998,0.995,0.993,0.985,0.967,0.933], 'SANTOS': [0.934,0.903,0.886,0.873,0.845,0.814], 'd3l': [0.807,0.804,0.8,0.792,0.777,0.765]}
    map_dict_tus_small = {'Starmie': 0.991, 'SingleCol': 0.954, 'SATO': 0.966, 'Sherlock': 0.984, 'SANTOS': 0.885, 'd3l': 0.794}
    recall_dict_tus_small = {'Starmie': [0.047,0.094,0.14,0.187,0.232,0.277], 'SingleCol': [0.046,0.091,0.135,0.177,0.217,0.255], 'SATO': [0.046,0.091,0.136,0.181,0.227,0.271], 'Sherlock': [0.047,0.094,0.141,0.186,0.229,0.265], 'SANTOS': [0.044,0.084,0.125,0.164,0.199,0.23], 'd3l': [0.038,0.076,0.113,0.149,0.182,0.215]}
    ideal_tus_small = [0.057,0.114,0.17,0.227,0.284,0.341]
    k_tus = [10,20,30,40,50,60]
    
    # plotJointFig(k_tus, 'tus_small', precision_dict_tus_small, recall_dict_tus_small, ideal_tus_small)
    # plotMapFig('tus_small', map_dict_tus_small)
    
    
    
    # ---------- 3. TUS Large Benchmark -----------
    precision_dict_tus_large = {'Starmie': [0.997,0.988,0.967,0.948,0.932,0.915], 'SingleCol': [0.951,0.925,0.903,0.876,0.85,0.824], 'SATO': [0.978,0.96,0.931,0.907,0.886,0.866], 'Sherlock': [0.929,0.83,0.734,0.654,0.581,0.525], 'd3l': [0.495,0.469,0.464,0.464,0.473,0.468]}
    map_dict_tus_large = {'Starmie': 0.965, 'SingleCol': 0.902, 'SATO': 0.93, 'Sherlock': 0.744, 'd3l': 0.484}
    recall_dict_tus_large = {'Starmie': [0.045,0.088,0.129,0.167,0.204,0.238], 'SingleCol': [0.043,0.082,0.119,0.153,0.183,0.208], 'SATO': [0.044,0.086,0.125,0.161,0.193,0.223], 'Sherlock': [0.041,0.071,0.092,0.105,0.114,0.119], 'd3l': [0.019,0.039,0.06,0.082,0.105,0.124]}
    ideal_tus_large = [0.046,0.092,0.138,0.185,0.231,0.277]
    
    # plotJointFig(k_tus, 'tus_large', precision_dict_tus_large, recall_dict_tus_large, ideal_tus_large)
    # plotMapFig('tus_large', map_dict_tus_large)
    
    
    
    ''' Plot scalability graphs for SANTOS REAL, WDC benchmarks. In the paper: include tables for indexing time and storage overhead '''
    # ====== Scalability Dictionaries ==========
    # ==========================================
    # ---------- 1. SANTOS Real Benchmark -----------
    scal_real_k = {'Linear': [71880,70620,70460,70540,70680,70580], 'Bounds': [30350,33050,34450,35520,36690,37230], 'LSH': [3470,3510,3460,3470,3420,3560], 'HNSW': [330,330,320,320,330,320]}
    scal_real_size = {'Linear': [13630,28120,42220,56920,70580], 'Bounds': [9540,16840,23930,30890,37230], 'LSH': [960,1590,2100,2890,3560], 'HNSW': [500,460,340,320,320]}
    k_scal = [10,20,30,40,50,60]
    dl_real_sizes = ['2.2K / 24K','4.4K / 48K','6.6K / 72K','8.8K / 96K','11K / 120K']
    plotScalFig(k_scal, dl_real_sizes, 'real', scal_real_k, scal_real_size)
    
    # ---------- 2. WDC Benchmark -----------
    scal_wdc_k = {'Linear': [865960,847880,818070,874010,874810,819170], 'Bounds': [341650,335850,356130,341370,339990,338580], 'LSH': [94370,101840,104660,94650,97420,106410], 'HNSW': [240,220,230,240,280,300]}
    scal_wdc_size = {'Linear': [161910,324000,488310,668493,819170], 'Bounds': [69530,155140,204780,274310,338580], 'LSH': [16350,34450,53850,72670,106410], 'HNSW': [230,290,200,340,300]}
    dl_wdc_sizes = ['200K / 1M ','400K / 2M','600K / 3M','800K / 4M','1M / 5M']
    # plotScalFig(k_scal, dl_wdc_sizes, 'wdc', scal_wdc_k, scal_wdc_size)