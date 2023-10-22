
import os
import subprocess  # to check if running at lipml

os.environ[ 'MPLCONFIGDIR' ] = '/tmp/$USER/'
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
from cycler import cycler
import scipy.stats as stats
from scipy import interpolate
import numpy as np
import argparse
import json
from copy import deepcopy



euclidean_distance = lambda x, y: np.sqrt(np.sum((x - y)**2, axis=1))
difference = lambda x, y: x - y

# FIXME ver com o prof pedro
difference_x = lambda x, y: (x[:, 0] - y[:, 0])
# difference_x = lambda x, y: np.abs(x[:, 0] - y[:, 0])
difference_y = lambda x, y: (x[:, 1] - y[:, 1])
difference_z = lambda x, y: (x[:, 2] - y[:, 2])
# euc_dist_norm = lambda x, y: .5 * np.var(x - y, axis=1) / (np.var(x, axis=1) + np.var(y, axis=1))

def assign_bins(results, jet_var, bins, mean=True):
    bins_values = []
    bins_values_std = []
    jet_bins = []
    for i in range(len(bins)):
        bins_values.append([])
        bins_values_std.append([])
    # Assign each jet value to a bin
    counter = 0
    for i in range(len(jet_var)):
        belongs_to_something = False
        for j in range(len(bins)-1):
            if j < len(bins) - 2 and jet_var[i] >= bins[j] and jet_var[i] < bins[j+1]:
                jet_bins.append(j)
                bins_values[j].append(results[i])
                belongs_to_something = True
                break
            elif j == len(bins) - 2 and jet_var[i] >= bins[j] and jet_var[i] <= bins[j+1]:
                jet_bins.append(j)
                bins_values[j].append(results[i])        
                belongs_to_something = True
        if not belongs_to_something:
            counter += 1

    if not mean:
        return bins_values

    # print(counter, "samples with no bin")
    for i in range(len(bins)-1):
        values_to_mean = bins_values[i]
        bins_values[i] = np.nanmean(values_to_mean)
        bins_values_std[i] = np.nanstd(values_to_mean)
        # bins_values[i] = np.mean(values_to_mean)
        if "nan" == str(bins_values[i]):
            # print(i, values_to_mean)
            # print("nan found")
            pass
    # print("BINS VALUES", bins_values)
    for i in range(len(bins)-2, 0, -1):
        bins_values.insert(i, bins_values[i-1])
        bins_values_std.insert(i, bins_values_std[i-1])
    bins_values[-1] = bins_values[-2]
    bins_values_std[-1] = bins_values_std[-2]
    return bins_values, bins_values_std



def plot_bins(predictions, targets, result_fn, jet_var, save_dir, name, colors, labels, bins, xlabel=None, ylabel=None, ylims=None, errors=None, show_errors=False):

    fig, ax = plt.subplots(1, figsize=(8,8))#, gridspec_kw={'height_ratios': [2, 1]})
    bins = list(bins)
    if ylims is None:
        y_max = -np.inf
        y_min = np.inf
    else:
        y_min, y_max = ylims

    bins_values_mean = []
    bins_values_std = []
    for t in range(len(predictions)):
        bins_values_aux = []
        bins_values_std_aux = []
        for inst in range(len(predictions[t])):
            results = result_fn(predictions[t][inst], targets[t])
            m, std = assign_bins(results, jet_var=jet_var[t], bins=bins)
            bins_values_aux.append(m)
            bins_values_std_aux.append(std)

        bins_values_aux = np.array(bins_values_aux)
        bins_values_std_aux = np.array(bins_values_std_aux)

        bins_values_mean.append(bins_values_aux.mean(axis=0))
        bins_values_std.append(bins_values_aux.std(axis=0))
        # bins_values_std.append(bins_values_std_aux.mean(axis=0))


    for i in range(len(bins)-2, 0, -1):
        bins.insert(i, bins[i])

    for x in range(len(bins_values_mean)):
        plt.plot(bins, bins_values_mean[x], label=labels[x], color=colors[x])#, color=colors[m])
        std_upper = bins_values_mean[x] + bins_values_std[x]
        std_lower = bins_values_mean[x] - bins_values_std[x]
        if show_errors:
            plt.fill_between(bins, std_lower, std_upper, color=colors[x], alpha=0.2)
        if ylims is None:
            y_max = max(y_max, 1.05 * max(bins_values_mean[x]))
            y_min = min(y_min, min(bins_values_mean[x]) - .05*abs(min(bins_values_mean[x])))  # FIXME

    ax.set_xlabel(xlabel)
    ax.set_xlim(min(bins), max(bins))

    ax.set_ylabel(ylabel)
    try:
        ax.set_ylim(y_min, y_max)
    except:
        ax.set_ylim(0, 11)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.pdf")
    plt.close()
    return  


def plot_fn(predictions, targets, result_fn, jet_var, save_dir, name, colors, labels, lims, xlabel=None, ylabel=None):
    # Dist. pred - true
    # fig, axes = plt.subplots(3,6,figsize=(15,9))
    mask = None
    __MAX = 25
    fig, ax = plt.subplots()

    ylim = -np.inf

    values_mean = []
    values_std = []
    for t in range(len(predictions)): # per model
        values_aux = []
        for inst in range(len(predictions[t])):
            results = result_fn(predictions[t][inst], targets[t])
            values_aux.append(results)
        values_aux = np.array(values_aux)
        values_mean.append(values_aux.mean(axis=0))
        values_std.append(values_aux.std(axis=0))

    lim1, lim2 = lims
    nbins = 200
    for x in range(len(values_mean)):
        plt.hist(
            values_mean[x],
            bins=np.linspace(lim1,lim2,nbins),
            histtype='step',
            linewidth=2,
            density=True,
            color=colors[x],
            label=labels[x]
        )
        # plt.plot(bins, values_mean[x], label=labels[x], color=colors[x])#, color=colors[m])
        # std_upper = values_mean[x] + values_std[x]
        # std_lower = values_mean[x] - values_std[x]
        # plt.fill_between(bins, std_lower, std_upper, color=colors[x], alpha=0.2)


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xlim(lim1, lim2)
    ax.legend() #loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{name}.pdf")
    plt.close()
 

def pretty_plots(name):
    cmap = get_cmap(name) 
    colors = cmap.colors  
    default_cycler = (cycler(color=colors))
    plt.rc('lines', linewidth=3)
    plt.rc('axes', prop_cycle=default_cycler)
    return(default_cycler)


def plots_3d_fitting(arr_true,arr_pred,arr_sigma,ds_name,save_dir, name):
    fig, axes = plt.subplots(3,6,figsize=(15,9))
    colors = pretty_plots("Dark2").by_key()['color']
    
    for dim, dimension in enumerate(["x","y","z"]):
        nbins = 50
        lim1 = -15
        lim2 = 15
        axes[dim][0].hist2d(arr_true[:,dim],arr_pred[:,dim],bins=(np.linspace(lim1,lim2,nbins),np.linspace(lim1,lim2,nbins)),cmap=plt.cm.jet,norm = mpl.colors.LogNorm())
        axes[dim][0].set_xlabel('true [mm]', fontsize=14)
        axes[dim][0].set_ylabel('pred [mm]', fontsize=14)
        axes[dim][0].set_xlim(lim1, lim2)
        axes[dim][0].set_ylim(lim1, lim2)
        axes[dim][0].plot([0, 1], [0, 1], transform=axes[dim][0].transAxes,c='grey', lw=1)
        
        lim1 = -5
        lim2 = 5
        axes[dim][1].hist(arr_pred[:,dim],bins=np.linspace(lim1,lim2,nbins),histtype='step',linewidth=2,color=colors[2],label="Pred")
        axes[dim][1].hist(arr_true[:,dim],bins=np.linspace(lim1,lim2,nbins),histtype='step',linewidth=2,color=colors[0],label="True")
        axes[dim][1].set_xlabel('%s [mm]'%dimension, fontsize=14)
        axes[dim][1].set_yscale('log')
        axes[dim][1].legend(loc='lower center')

        if arr_sigma is not None:
            nbins = 50
            axes[dim][2].hist(arr_sigma[:,dim].flatten(),bins=np.linspace(0,200,nbins),histtype='step',linewidth=2,color=colors[2],label="Pred")
            axes[dim][2].set_xlabel(r'$\sigma_%s$ [mm]'%dimension, fontsize=14)
            axes[dim][2].set_ylabel('# Jets', fontsize=14)
            axes[dim][2].set_yscale('log')
            axes[dim][2].legend()
            
        if arr_sigma is not None:

            nbins = 50
            lim1 = -20
            lim2 = 20
            m = (arr_pred[:,dim]-arr_true[:,dim])/arr_sigma[:,dim]
            axes[dim][3].hist(m,bins=np.linspace(lim1,lim2,nbins),histtype='step',linewidth=2,color=colors[2],label="Pred")
            axes[dim][3].set_xlabel(r'$(%s_{pred}-%s_{true}) / \sigma$'%(dimension,dimension), fontsize=14)
            axes[dim][3].set_yscale('log')
            axes[dim][3].legend()
            # Check
            goodenough = abs(arr_pred[:,dim]-arr_true[:,dim]) < 5
            ratio = 100*(np.sum(goodenough)/len(arr_true[:,dim]))
            print("Percentage with pred-true difference below +/-5 mm: %0.2f"%ratio)
            goodenough = abs(arr_pred[:,dim]-arr_true[:,dim]) < arr_sigma[:,dim]
            ratio = 100*(np.sum(goodenough)/len(arr_true[:,dim]))
            print("Percentage with pred-true difference below +/- sigma mm: %0.2f"%ratio)
    
        if arr_sigma is not None:

            lim1 = -5
            lim2 = 5
            axes[dim][4].hist2d(m,arr_true[:,dim],bins=(np.linspace(lim1,lim2,nbins),np.linspace(lim1,lim2,nbins)),cmap=plt.cm.jet,norm = mpl.colors.LogNorm())
            axes[dim][4].set_xlabel(r'$(%s_{pred}-%s_{true}) / \sigma$'%(dimension,dimension), fontsize=14)
            axes[dim][4].set_ylabel('true [mm]', fontsize=14)
        
        if arr_sigma is not None:

            lim1 = -5
            lim2 = 5
            delta = (arr_pred[:,dim]-arr_true[:,dim])
            axes[dim][5].hist2d(delta,arr_sigma[:,dim],bins=(np.linspace(lim1,lim2,nbins),np.linspace(0,lim2,nbins)),cmap=plt.cm.jet,norm = mpl.colors.LogNorm())
            axes[dim][5].set_xlabel(r'$%s_{pred}-%s_{true}$ [mm]'%(dimension,dimension), fontsize=14)
            axes[dim][5].set_ylabel('$\sigma$ [mm]', fontsize=14)

    plt.suptitle("%s"%ds_name, fontsize=18)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/{name}.pdf") 
    return


# Higher-level functions
def plot_global_performance(predictions, errors, true_vtx, true_graph, jet_var, jet_var_label, labels, var_dir, bins):
    with open("configs.json", "r") as f:
        settings = json.load(f)
        colors = settings['colors']
        for c in range(len(colors)):
            if colors[c][0] == "(":
                colors[c] = eval(colors[c])
        results_dir = settings['results_dir'] + "/" + var_dir

    plot_bins(
        predictions, 
        [true_vtx] * len(predictions), 
        euclidean_distance,
        [jet_var] * len(predictions),
        results_dir,
        "bins_euclidean_distance", 
        colors,
        labels,
        bins, 
        xlabel=jet_var_label, #r"Jet $p_{T}$ [GeV]",
        ylabel="Mean Euclidean Distance [mm]"
    )

    labels_axs = [r"$x_{pred} - x_{true}$", r"$y_{pred} - y_{true}$", r"$z_{pred} - z_{true}$"]
    names_axs = ['x', 'y', 'z']
    fn_axs = (difference_x, difference_y, difference_z)
    for dimension in range(len(names_axs)):
        plot_fn(
            predictions, 
            [true_vtx] * len(predictions), 
            fn_axs[dimension],
            [jet_var] * len(predictions),
            results_dir,
            "difference_" + names_axs[dimension], 
            colors,
            labels,
            (-5, 5),
            xlabel=labels_axs[dimension] + " [mm]",
            ylabel=""
        )


def plot_fitting_average(predictions, errors, true_vtx, true_graph, jet_var, labels, bins):
    with open("configs.json", "r") as f:
        settings = json.load(f)
        results_dir = settings['results_dir']

    predictions = np.array(predictions)
    predictions = predictions.mean(axis=1)
    predictions = list(predictions) # |models| x |test_dl| x 3

    errors = np.array(errors)
    errors = errors.mean(axis=1)
    errors = list(errors) # |models| x |test_dl| x 3

    flavour_set = np.unique(true_graph).tolist()

    """
    for m in range(len(predictions)):
        pred_bins = assign_bins(predictions[m], jet_var, bins, mean=False)
        errors_bins = assign_bins(errors[m], jet_var, bins, mean=False)
        true_vtx_bins = assign_bins(true_vtx, jet_var, bins, mean=False)
        true_graph_bins = assign_bins(true_graph, jet_var, bins, mean=False)

        for i in range(1, len(bins)-1):
            for flavour in flavour_set:
                true_vtx_bins[i] = np.array(true_vtx_bins[i])
                pred_bins[i] = np.array(pred_bins[i])
                errors_bins[i] = np.array(errors_bins[i])
                true_graph_bins[i] = np.array(true_graph_bins[i])

                plots_3d_fitting(
                    true_vtx_bins[i][true_graph_bins[i] == flavour], 
                    pred_bins[i][true_graph_bins[i] == flavour], 
                    errors_bins[i][true_graph_bins[i] == flavour], 
                    "Test dataset", 
                    results_dir,
                    f"fitting_flavour_bin_{i}_flavour_{flavour}_{labels[m]}"
                )    
    """
    for m in range(len(predictions)):
        plots_3d_fitting(true_vtx, predictions[m], errors[m], "Test dataset", results_dir, f"fitting_{labels[m]}")
        for flavour in flavour_set:
            plots_3d_fitting(
                true_vtx[true_graph == flavour], 
                predictions[m][true_graph == flavour], 
                errors[m][true_graph == flavour], 
                "Test dataset", 
                results_dir,
                f"fitting_flavour_{flavour}_{labels[m]}"
            )



def plot_discriminated_by_flavour(predictions, errors, true_vtx, true_graph, jet_var, jet_var_label, labels, var_dir, bins):
    with open("configs.json", "r") as f:
        settings = json.load(f)
        results_dir = settings['results_dir'] + "/" + var_dir
        colors = settings['colors_flavours']
        for c in range(len(colors)):
            if colors[c][0] == "(":
                colors[c] = eval(colors[c])
        labels_flav= settings['labels_graph']

    flavour_set = np.unique(true_graph)

    true_vtx_by_flavour = []
    jet_var_by_flavour = []
    for flavour in flavour_set:
        true_vtx_by_flavour.append(true_vtx[flavour == true_graph])
        jet_var_by_flavour.append(jet_var[flavour == true_graph])

    for m in range(len(predictions)):
        predictions_by_flavour = []
        errors_by_flavour = []
        prediction_over_errors = []
        true_vtx_over_errors = []

        for flavour in flavour_set:
            predictions_by_flavour.append([])
            errors_by_flavour.append([])
            prediction_over_errors.append([])
            
            for i in range(len(predictions[m])):
                predictions_by_flavour[-1].append(predictions[m][i][flavour == true_graph])
                errors_by_flavour[-1].append(errors[m][i][flavour == true_graph])
                true_vtx_over_errors.append(predictions[m][i][flavour == true_graph] / errors[m][i][flavour == true_graph])
                prediction_over_errors[-1].append(true_vtx[flavour == true_graph] / errors[m][i][flavour == true_graph])
        
        plot_bins(
            predictions_by_flavour, 
            true_vtx_by_flavour, 
            euclidean_distance,
            jet_var_by_flavour,
            results_dir,
            f"bins_euclidean_distance_{labels[m]}", 
            colors,
            labels_flav,
            bins, 
            xlabel=jet_var_label,
            ylabel="Mean Euclidean Distance [mm]",
            ylims=(0.0, 8.0),
            show_errors=True
        )

        plot_fn(
            predictions_by_flavour, 
            true_vtx_by_flavour, 
            euclidean_distance,
            jet_var_by_flavour,
            results_dir,
            f"euclidean_distance_{labels[m]}",
            colors,
            labels_flav,
            (0, 50),
            xlabel= "Euclidean distance [mm]",
            ylabel="#Jets"
        )

        labels_axs = [
            r"$(x_{pred} - x_{true}) / \sigma_x$", 
            r"$(y_{pred} - y_{true}) / \sigma_y$", 
            r"$(z_{pred} - z_{true}) / \sigma_z$"
        ]
        names_axs = ['x', 'y', 'z']
        fn_axs = (difference_x, difference_y, difference_z)
        for dimension in range(len(names_axs)):
            plot_fn(
                prediction_over_errors, 
                true_vtx_over_errors, 
                fn_axs[dimension],
                jet_var_by_flavour,
                results_dir,
                f"difference_over_error_{names_axs[dimension]}_{labels[m]}", 
                colors,
                labels_flav,
                (-5, 5),
                xlabel=labels_axs[dimension] + " [mm]",
                ylabel=""
            )
        
            plot_bins(
                prediction_over_errors, 
                true_vtx_over_errors, 
                fn_axs[dimension],
                jet_var_by_flavour,
                results_dir,
                f"bins_difference_over_error_{names_axs[dimension]}_{labels[m]}", 
                colors,
                labels_flav,
                bins, 
                xlabel=jet_var_label,
                ylabel=labels_axs[dimension] + " [mm]",
                ylims=(-3.0, 3.0)
            )

def performance_regression(labels, predictions, errors, targets, flavours, jet_pts, jet_trks):
    print("Evaluating regression")
    try:
        print('Shapes: (targets, flavours, jet_pts) =', (targets.shape, flavours.shape, jet_pts.shape))


        # flavours = flavours.tolist() * len(predictions)
        labels_flav = ['b', 'c', 'u']
        # |predictions| =|models| x |ensemble| x |test_dl| x 3
        # -> |jets| x |ensemble| x |test_dl| x 3
        print("Per model distance plots")
        for m, preds in enumerate(predictions):    
            p = []
            t, jpts, jtrks = [], [], []
            for flavour in flavour_set:
                p.append([])
                for inst in range(len(preds)):
                    p[-1].append(preds[inst][flavours == flavour])
                t.append(targets[flavours == flavour])
                jpts.append(jet_pts[flavours == flavour])
                # jtrks.append(jet_trks[flavours == flavour])
            assert(len(p) == len(flavour_set))
            assert(all(len(p[x]) == len(preds) for x in range(len(p))))

            plot_bins(
                p, 
                t, 
                euc_dist,
                jpts,
                IMG_DIR + f"bins_euclidean_distance_{labels[m]}", 
                colors,
                labels_flav,
                bins, 
                xlabel=r"Jet $p_{T}$ [GeV]",
                ylabel="Mean Euclidean Distance [mm]",
                ylims=(0.0, 8.0)
            )

            plot_bins(
                p, 
                t, 
                euc_dist_norm,
                jpts,
                IMG_DIR + f"bins_norm_euclidean_distance_{labels[m]}", 
                colors,
                labels_flav,
                bins, 
                xlabel=r"Jet $p_{T}$ [GeV]",
                ylabel="Normalised Mean Euclidean Distance [mm]"
            )

            for ttl, diff in enumerate((difference_x, difference_y, difference_z)):
                plot_fn(
                    p, 
                    t, 
                    diff,
                    jpts,
                    IMG_DIR + f"difference_{labels[m]}_{ttl2[ttl]}", 
                    colors,
                    labels_flav,
                    (-5, 5),
                    xlabel=r"Jet $p_{T}$ [GeV]",
                    ylabel=ttls[ttl] + " [mm]"
                )

            plot_fn(
                p, 
                t, 
                euc_dist,
                jpts,
                IMG_DIR + f"euclidean_distance_{labels[m]}",
                colors,
                labels_flav,
                (0, 25),
                xlabel= "Euclidean distance [mm]",
                ylabel="#Jets"
            )

        print("ok")
        # for flavour in flavour_set:
    except Exception as e:
        print("Evaluating regression: FAILED")
        print(e)
        raise e
        exit(0)

