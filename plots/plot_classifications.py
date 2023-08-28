import os
import subprocess  # to check if running at lipml

os.environ[ 'MPLCONFIGDIR' ] = '/tmp/$USER/'
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import interpolate
import numpy as np
import argparse
import json

font = {'family' : 'sans-serif',
        'size'   : 16}
plt.rc('font', **font)

# Auxiliary functions
def plot_confusion_matrix(predictions, targets, classes, name, save_dir):
    plt.figure(figsize=(4,4))
    
    predictions = np.argmax(predictions, axis=1) 
    conf_matrix = confusion_matrix(targets, predictions, normalize="true", labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{name}.pdf")
    plt.close()


def plot_standard_roc(predictions, targets, classes, name, save_dir):
    plt.figure(figsize=(7,7))

    for pos_class in range(len(classes)):
        fpr_node, tpr_node, _ = roc_curve(targets, predictions[:, pos_class], pos_label=pos_class)
        roc_auc = auc(fpr_node, tpr_node)
        plt.plot(fpr_node,tpr_node,lw=2, label="ROC curve class %d (area = %0.2f)" % (pos_class,roc_auc))

    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: " + name + " classification")
    plt.legend(loc="lower right")

    plt.savefig(f"{save_dir}/standard_roc_{name}.pdf")
    plt.close()


def prepare_ROC_jet_ATLAS_bcujets(pred, true, f=0.05, rej_threshold=100.0):
    true = np.array(true) # .cpu()
    pb = pred[:,0]
    pc = pred[:,1]
    pu = pred[:,2]
    
    denom = (1-f)*pu + f*pc

    valid1 = ( denom != 0 ) 
    valid2 = ( pb != 0 ) & ( np.isfinite(pb) )
    discriminator = np.empty_like(pb)

    discriminator[valid1&valid2] = np.log( np.divide(pb[valid1&valid2],denom[valid1&valid2]))
    maxval = np.max(discriminator[valid1&valid2])
    minval = np.min(discriminator[valid1&valid2])

    discriminator[~valid1] = maxval
    discriminator[~valid2] = minval

    bjet = discriminator[true==0]
    cjet = discriminator[true==1]
    ujet = discriminator[true==2]

    plt.figure(figsize=(7,7))
    n_c, bins_c, _ = plt.hist(cjet, bins=np.linspace(minval,maxval,200), histtype="step", lw=2, label="c-jets", density=True)
    n_u, bins_u, _ = plt.hist(ujet, bins=np.linspace(minval,maxval,200), histtype="step", lw=2, label="u-jets", density=True)
    n_b, bins_b, _ = plt.hist(bjet, bins=np.linspace(minval,maxval,200), histtype="step", lw=3, label="b-jets", density=True)
    # plt.yscale("log")
    # plt.xlabel("Discriminator")
    # plt.ylabel("Arbitrary Units")
    # plt.legend(loc="lower right")

    # plt.title(name.replace("_"," ")) #+" f = %0.3f"%f)
    # plt.savefig(f"{img_dir}discriminator_{name}.{extension}")
    # plt.close()

    # plt.figure(figsize=(7,7))
    # plt.hist(pb[true==2], bins=np.linspace(0,1,25), color="blue", histtype="step", lw=1, label="pb b-jets", density=True)
    # plt.hist(pb[true!=2], bins=np.linspace(0,1,25), color="blue", linestyle="--", histtype="step", lw=1, label="pb non b-jets", density=True)
    # plt.hist(pc[true==1], bins=np.linspace(0,1,25), color="red", histtype="step", lw=1, label="pc c-jets", density=True)
    # plt.hist(pc[true!=1], bins=np.linspace(0,1,25), color="red", linestyle="--", histtype="step", lw=1, label="pc non c-jets", density=True)
    # plt.hist(pu[true==0], bins=np.linspace(0,1,25), color="green", histtype="step", lw=1, label="pu u-jets", density=True)
    # plt.hist(pu[true!=0], bins=np.linspace(0,1,25), color="green", linestyle="--", histtype="step", lw=1, label="pu non u-jets", density=True)
    # plt.yscale("log")
    # plt.xlabel("Probabilities")
    # plt.ylabel("Arbitrary Units")
    # plt.legend(loc="lower right")
    # plt.title(name.replace("_"," ")+" f = %0.1f"%f)
    # plt.savefig(f"{img_dir}probabilities_{name}.{extension}") 
    # plt.close()

    # signal efficiency 
    sig_eff_num = np.cumsum(n_b[::-1])[::-1]
    sig_eff_denom = sig_eff_num.max()
    
    sig_eff = sig_eff_num/sig_eff_denom
    valid_sig = sig_eff > .2
    # bkg rejection for charm jets
    cbkg_num = np.cumsum(n_c[::-1])[::-1] 
    cbkg_denom = cbkg_num.max()

    cbkg_eff = cbkg_num/cbkg_denom
    cbkg_rej = np.zeros_like(cbkg_eff)
    valid = cbkg_eff > 0.0
    cbkg_rej[valid] = 1/cbkg_eff[valid]
    # bkg rejection for light jets
    ubkg_num = np.cumsum(n_u[::-1])[::-1] 
    ubkg_denom = ubkg_num.max()
    
    ubkg_eff = ubkg_num/ubkg_denom
    ubkg_rej = np.zeros_like(ubkg_eff)
    valid = ubkg_eff > 0.0
    ubkg_rej[valid] = 1/ubkg_eff[valid]

    closest_idx = 0
    for i in range(len(ubkg_rej)):
        if ubkg_rej[i] > rej_threshold:
            closest_idx += 1

    discriminator_score = bins_u[closest_idx]  # REJECTION OF U-JETS = 100
    return (sig_eff[valid_sig], cbkg_rej[valid_sig], ubkg_rej[valid_sig])


def compare_models_jet_ATLAS(jet_results, colors, save_dir, labels):
    fig, ax = plt.subplots(2, 2 ,figsize=(16,8),gridspec_kw={'height_ratios': [2, 1]})

    # Upper panel (ROC)
    sig_eff_allmeans = []
    bkg_crej_allmeans = []
    bkg_crej_allstds = []
    bkg_urej_allmeans = []
    bkg_urej_allstds = []

    for m, model_results in enumerate(jet_results):
        mean_sig_eff = np.linspace(0, 1, 1000)
        interp_cbkg_rej = []
        interp_ubkg_rej = []
        for i, sig_bkg in enumerate(model_results): # each sample of the model
            sig_eff  = sig_bkg[0]
            cbkg_rej = sig_bkg[1]
            ubkg_rej = sig_bkg[2]
            interp_c = interpolate.interp1d(sig_eff,cbkg_rej,bounds_error=False)
            interp_cbkg_rej.append(interp_c(mean_sig_eff))
            interp_u = interpolate.interp1d(sig_eff,ubkg_rej,bounds_error=False)
            interp_ubkg_rej.append(interp_u(mean_sig_eff))
        mean_cbkg_rej = np.mean(interp_cbkg_rej,axis=0)
        mean_ubkg_rej = np.mean(interp_ubkg_rej,axis=0)
        ax[0, 0].plot(mean_sig_eff,mean_cbkg_rej,color=colors[m],lw=2, label=labels[m])
        ax[0, 1].plot(mean_sig_eff,mean_ubkg_rej,color=colors[m],lw=2, label=labels[m])

        bkg_crej_allmeans.append(mean_cbkg_rej)
        bkg_urej_allmeans.append(mean_ubkg_rej)
        sig_eff_allmeans.append(mean_sig_eff)

        std_rej = np.std(interp_cbkg_rej, axis=0)
        bkg_crej_allstds.append(std_rej)
        rej_upper = mean_cbkg_rej + std_rej
        rej_lower = mean_cbkg_rej - std_rej
        ax[0, 0].fill_between( mean_sig_eff, rej_lower, rej_upper, color=colors[m], alpha=0.2)
        
        std_rej = np.std(interp_ubkg_rej, axis=0)
        bkg_urej_allstds.append(std_rej)
        rej_upper = mean_ubkg_rej + std_rej
        rej_lower = mean_ubkg_rej - std_rej
        ax[0, 1].fill_between(mean_sig_eff, rej_lower, rej_upper, color=colors[m], alpha=0.2)
        
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_ylabel("c-jet rejection", loc="top")
    ax[0, 0].legend(loc="upper right")
    ax[0, 0].set_xlim(0.6, 1)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel("u-jet rejection", loc="top")
    ax[0, 1].legend(loc="upper right")
    ax[0, 1].set_xlim(0.6, 1)

    # Ratio panel
    # let's assume denominator is always first model
    baseline_sig_eff, baseline_cbkg_rej, baseline_ubkg_rej = sig_eff_allmeans[0], bkg_crej_allmeans[0], bkg_urej_allmeans[0]
    cbkg_rej_finder = interpolate.interp1d(baseline_sig_eff,baseline_cbkg_rej,bounds_error=False)
    ubkg_rej_finder = interpolate.interp1d(baseline_sig_eff,baseline_ubkg_rej,bounds_error=False)
    for r in range(len(jet_results)):
        num_sig_eff, num_cbkg_rej, num_ubkg_rej = sig_eff_allmeans[r], bkg_crej_allmeans[r], bkg_urej_allmeans[r]
        c_interp_baseline_rej = cbkg_rej_finder(num_sig_eff)
        u_interp_baseline_rej = ubkg_rej_finder(num_sig_eff)
        
        c_ratio_of_rejections = np.ones_like(c_interp_baseline_rej)
        c_ratio_of_rejections = num_cbkg_rej/c_interp_baseline_rej
        cjet, = ax[1, 0].plot(num_sig_eff,c_ratio_of_rejections, color='black', lw=2)
        ax[1, 0].plot(num_sig_eff,c_ratio_of_rejections, color=colors[r], lw=2)


        u_ratio_of_rejections = np.ones_like(u_interp_baseline_rej)
        u_ratio_of_rejections = num_ubkg_rej/u_interp_baseline_rej
        ujet, = ax[1, 1].plot(num_sig_eff,u_ratio_of_rejections, color='black', lw=2)
        ax[1, 1].plot(num_sig_eff,u_ratio_of_rejections, color=colors[r], lw=2)
    
        tot_c_err = c_ratio_of_rejections * np.sqrt(
            np.square(bkg_crej_allstds[r]/num_cbkg_rej)+np.square(bkg_crej_allstds[0]/baseline_cbkg_rej)
        )
        tot_u_err = u_ratio_of_rejections * np.sqrt(
            np.square(bkg_urej_allstds[r]/num_ubkg_rej)+np.square(bkg_urej_allstds[0]/baseline_ubkg_rej)
        )

        ax[1, 0].fill_between(num_sig_eff, c_ratio_of_rejections - tot_c_err, c_ratio_of_rejections + tot_c_err, color=colors[r], alpha=0.2)
        ax[1, 1].fill_between(num_sig_eff, u_ratio_of_rejections - tot_u_err, u_ratio_of_rejections + tot_u_err, color=colors[r], alpha=0.2)

    ax[1, 0].set_xlabel("b-jet efficiency")
    ax[1, 0].set_ylabel("Ratio to " + labels[0])
    ax[1, 0].set_xlim(.6, 1)
    ax[1, 1].set_xlabel("b-jet efficiency")
    ax[1, 1].set_ylabel("Ratio to " + labels[0])
    ax[1, 1].set_xlim(.6, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ATLAS_roc_ensemble_bcu.pdf")
    plt.close()
    
    return    


# High-level functions
def plot_classifications(predictions, targets, labels, labels_key):
    predictions = np.array(predictions)
    predictions = predictions.mean(axis=1)
    predictions = list(predictions)
    with open("configs.json", "r") as f:
        settings = json.load(f)
        clf_type = labels_key
        labels_key = 'labels_' + labels_key
        if labels_key not in settings.keys():
            raise ValueError("Unexistent classification labels.")
        results_dir = settings['results_dir']
        labels_clf = settings[labels_key]
    for t in range(len(predictions)):
        try:
            plot_confusion_matrix(predictions[t], targets, labels_clf, clf_type + "_" + labels[t], results_dir)
        except Exception:
            print("Confusion matrix not plotted.")
        try:
            plot_standard_roc(predictions[t], targets, labels_clf, clf_type + "_" + labels[t], results_dir)
        except Exception:
            print("Standard ROC not plotted.")



def performance_roc_jets(predictions, true_flavours, labels):

    results = []
    for t in range(len(predictions)):
        model_results = []
        for inst in range(len(predictions[t])):
            data = prepare_ROC_jet_ATLAS_bcujets(predictions[t][inst], true_flavours)
            model_results.append(list(data))
            
        results.append(model_results)

    with open("configs.json", "r") as f:
        settings = json.load(f)
        results_dir = settings['results_dir']
        colors = settings['colors']

    compare_models_jet_ATLAS(results, colors, results_dir, labels)

