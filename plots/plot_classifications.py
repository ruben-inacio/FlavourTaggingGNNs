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
from plot_vertex_fitting import assign_bins
import copy

def reset_font():
    font = {'family' : 'sans-serif',
            'size'   : 16}
    plt.rc('font', **font)

reset_font()

# Auxiliary functions
def plot_confusion_matrix(predictions, targets, classes, name, save_dir):
    plt.figure(figsize=(4,4))
    
    predictions = np.argmax(predictions, axis=1) 
    conf_matrix = confusion_matrix(targets, predictions, normalize="true", labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(cmap='Blues')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix_{name}.pdf")
    plt.close()


def plot_standard_roc(predictions, targets, classes, name, save_dir):
    font = {'family' : 'sans-serif',
        'size'   : 22}
    plt.rc('font', **font)
    plt.figure(figsize=(7,7))
    plt.subplots_adjust(left=0.14, right=.985, top=.925, bottom=0.125)

    for pos_class in range(len(classes)):
        fpr_node, tpr_node, _ = roc_curve(targets, predictions[:, pos_class], pos_label=pos_class)
        roc_auc = auc(fpr_node, tpr_node)
        plt.plot(fpr_node,tpr_node,lw=3, label="%s (AUC = %0.3f)" % (classes[pos_class],roc_auc))

    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Model: " + name.split("_")[-1])
    # plt.title("ROC: " + name + " classification")
    plt.legend(loc="lower right")

    plt.savefig(f"{save_dir}/standard_roc_{name}.pdf")
    plt.close()
    reset_font()


def prepare_ROC_jet_ATLAS_bcujets(pred, true, f=0.05, rej_threshold=100.0):
    true = np.array(true) # .cpu()
    pb = pred[:,0]
    pc = pred[:,1]
    pu = pred[:,2]
    
    denom = (1-f)*pu + f*pc

    valid1 = ( denom != 0 ) 
    valid2 = ( pb != 0 ) & ( np.isfinite(pb) )
    discriminator = np.empty_like(pb)

    discriminator[valid1&valid2] = np.log(np.divide(pb[valid1&valid2],denom[valid1&valid2]))
    maxval = np.max(discriminator[valid1&valid2])
    minval = np.min(discriminator[valid1&valid2])

    discriminator[~valid1] = maxval
    discriminator[~valid2] = minval

    bjet = discriminator[true==0]
    cjet = discriminator[true==1]
    ujet = discriminator[true==2]

    plt.figure(figsize=(7,7))
    n_c, bins_c, _ = plt.hist(cjet, bins=np.linspace(minval,maxval,200), histtype="step", lw=2, label="c-jets", density=True)
    n_u, bins_u, _ = plt.hist(ujet, bins=np.linspace(minval,maxval,200), histtype="step", lw=2, label="l-jets", density=True)
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
    # plt.hist(pu[true==0], bins=np.linspace(0,1,25), color="green", histtype="step", lw=1, label="pu l-jets", density=True)
    # plt.hist(pu[true!=0], bins=np.linspace(0,1,25), color="green", linestyle="--", histtype="step", lw=1, label="pu non l-jets", density=True)
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
    valid_sig = sig_eff > 0.575
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

    discriminator_score = bins_u[closest_idx]  # REJECTION OF l-JETS = 100
    return (sig_eff[valid_sig], cbkg_rej[valid_sig], ubkg_rej[valid_sig])


def compare_models_jet_ATLAS(jet_results, colors, linestyles, markers, save_dir, labels):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    font = {'family' : 'sans-serif',
        'size'   : 18}
    plt.rc('font', **font)
    fig, ax = plt.subplots(2, 2 ,figsize=(16,8),gridspec_kw={'height_ratios': [2, 1]})
    # markevery = 25
    lw = 3
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
        ax[0, 0].plot(mean_sig_eff,mean_cbkg_rej,color=colors[m],lw=lw, label=labels[m], linestyle=linestyles[m])
        ax[0, 1].plot(mean_sig_eff,mean_ubkg_rej,color=colors[m],lw=lw, label=labels[m], linestyle=linestyles[m])

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
    ax[0, 0].set_ylabel(r"c-jet rejection ($1/\epsilon_c$)", loc="top")
    ax[0, 0].legend(loc="lower left", handlelength=3)
    # ax[0, 0].legend(loc="upper right", handlelength=3)
    ax[0, 0].set_xlim(0.6, 1)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_ylabel(r"l-jet rejection ($1/\epsilon_l$)", loc="top")
    ax[0, 1].legend(loc="lower left", handlelength=3)
    ax[0, 1].set_xlim(0.6, 1)

    # Ratio panel
    # let's assume denominator is always first model
    baseline_sig_eff, baseline_cbkg_rej, baseline_ubkg_rej = sig_eff_allmeans[0], bkg_crej_allmeans[0], bkg_urej_allmeans[0]
    cbkg_rej_finder = interpolate.interp1d(baseline_sig_eff,baseline_cbkg_rej,bounds_error=False)
    ubkg_rej_finder = interpolate.interp1d(baseline_sig_eff,baseline_ubkg_rej,bounds_error=False)
    c_rej_70 = []
    c_rej_85 = []
    u_rej_70 = []
    u_rej_85 = []
    for r in range(len(jet_results)):
        num_sig_eff, num_cbkg_rej, num_ubkg_rej = sig_eff_allmeans[r], bkg_crej_allmeans[r], bkg_urej_allmeans[r]
        idx_70 = find_nearest(num_sig_eff, 0.7)
        idx_85 = find_nearest(num_sig_eff, 0.85)

        c_interp_baseline_rej = cbkg_rej_finder(num_sig_eff)
        u_interp_baseline_rej = ubkg_rej_finder(num_sig_eff)
        
        c_ratio_of_rejections = np.ones_like(c_interp_baseline_rej)
        c_ratio_of_rejections = num_cbkg_rej/c_interp_baseline_rej
        # cjet, = ax[1, 0].plot(num_sig_eff,c_ratio_of_rejections, color='black', lw=2)
        ax[1, 0].plot(num_sig_eff,c_ratio_of_rejections, color=colors[r], lw=lw, linestyle=linestyles[r])


        u_ratio_of_rejections = np.ones_like(u_interp_baseline_rej)
        u_ratio_of_rejections = num_ubkg_rej/u_interp_baseline_rej
        # ujet, = ax[1, 1].plot(num_sig_eff,u_ratio_of_rejections, color='black', lw=2)
        ax[1, 1].plot(num_sig_eff,u_ratio_of_rejections, color=colors[r], lw=lw, linestyle=linestyles[r])
    
        c_rej_70.append(c_ratio_of_rejections[idx_70])
        c_rej_85.append(c_ratio_of_rejections[idx_85])
        u_rej_70.append(u_ratio_of_rejections[idx_70])
        u_rej_85.append(u_ratio_of_rejections[idx_85])
        tot_c_err = c_ratio_of_rejections * np.sqrt(
            np.square(bkg_crej_allstds[r]/num_cbkg_rej)+np.square(bkg_crej_allstds[0]/baseline_cbkg_rej)
        )
        tot_u_err = u_ratio_of_rejections * np.sqrt(
            np.square(bkg_urej_allstds[r]/num_ubkg_rej)+np.square(bkg_urej_allstds[0]/baseline_ubkg_rej)
        )

        ax[1, 0].fill_between(num_sig_eff, c_ratio_of_rejections - tot_c_err, c_ratio_of_rejections + tot_c_err, color=colors[r], alpha=0.2)
        ax[1, 1].fill_between(num_sig_eff, u_ratio_of_rejections - tot_u_err, u_ratio_of_rejections + tot_u_err, color=colors[r], alpha=0.2)

    ax[1, 0].set_xlabel(r"b-jet efficiency ($\epsilon_b$)")
    ax[1, 0].set_ylabel("Ratio to " + labels[0])
    ax[1, 0].set_xlim(.6, 1)
    ax[1, 1].set_xlabel(r"b-jet efficiency ($\epsilon_b$)")
    ax[1, 1].set_ylabel("Ratio to " + labels[0])
    ax[1, 1].set_xlim(.6, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ATLAS_roc_ensemble_bcu.pdf")
    plt.close()
    reset_font()
    with open(f"{save_dir}/ATLAS_roc_ensemble_wps.txt", "w") as fq:
        fq.write(f" c70 {str(c_rej_70)}\n c85 {str(c_rej_85)}\n u70 {str(u_rej_70)}\n u85 {str(u_rej_85)}")
    return    


def compute_threshold_rej(jet_flavours, discriminator, rejection_threshold=45.0, rejection_type='u'):
    if type(jet_flavours) != np.ndarray:
        jet_flavours = np.array(jet_flavours)
    if type(discriminator) != np.ndarray:
        discriminator = np.array(discriminator)
    # print(discriminator.shape)

    cjet = discriminator[jet_flavours==1]
    ujet = discriminator[jet_flavours==2]
    values = cjet if rejection_type == "c" else ujet

    n, bins = np.histogram(values, bins=np.linspace(np.min(discriminator),np.max(discriminator),200), density=True)
    
    # Compute rejection
    bkg_num = np.cumsum(n[::-1])[::-1] 
    bkg_denom = bkg_num.max()
    bkg_eff = bkg_num / bkg_denom
    bkg_rej = np.zeros_like(bkg_eff)
    valid = bkg_eff > 0.0
    bkg_rej[valid] = 1 / bkg_eff[valid]  # should be: 1/eps = x <=> 1 = x*eps <=> 1 - eps = (x-1)*eps?

    # closest_idx = 0
    # for i in range(len(bkg_rej)):
    #     if (bkg_rej[i] - 1) * rejection_threshold > 1 - rejection_threshold:
    #         closest_idx += 1
    bins = bins[:-1]
    scores = interpolate.interp1d(
        bkg_rej,
        bins,
        bounds_error=False
    )

    return scores(rejection_threshold)


def compute_threshold_eff(jet_flavours, discriminator, working_point):
    if type(jet_flavours) != np.ndarray:
        jet_flavours = np.array(jet_flavours)
    if type(discriminator) != np.ndarray:
        discriminator = np.array(discriminator)
    # print(discriminator.shape)

    values = discriminator[jet_flavours == 0]
    n, bins = np.histogram(values, bins=np.linspace(np.min(values),np.max(values),200), density=True)
    # n, bins = np.histogram(values, bins=np.linspace(np.min(discriminator),np.max(discriminator),200), density=True)

    
    # Compute rejection
    sig_num = np.cumsum(n[::-1])[::-1] 
    sig_denom = sig_num.max()
    sig_eff = sig_num / sig_denom
    
    bins = bins[:-1]
    scores = interpolate.interp1d(
        sig_eff,
        bins,
        bounds_error=False
    )

    return scores(working_point)


def get_discriminator(results):
    f = 0.05
    results = np.array(results)
    # print(results.shape)
    pb = np.array(results[:,0],dtype=np.float128)
    pc = np.array(results[:,1],dtype=np.float128)
    pu = np.array(results[:,2],dtype=np.float128)

    denom = (1-f)*pu + f*pc
    valid1 = ( denom != 0 ) 
    valid2 = ( pb != 0 ) & ( np.isfinite(pb) )
    discriminator = np.empty_like(pb)
    discriminator[valid1&valid2] = np.log( np.divide(pb[valid1&valid2],denom[valid1&valid2]))
    maxval = np.max(discriminator[valid1&valid2])
    minval = np.min(discriminator[valid1&valid2])
    discriminator[~valid1] = maxval
    discriminator[~valid2] = minval

    return discriminator, pb, pc, pu


def compare_models_eff_ATLAS(jet_results, colors, linestyles, markers, save_dir, labels, jet_var, jet_true, bins, var, var_label):
    font = {'family' : 'sans-serif',
        'size'   : 20}
    plt.rc('font', **font)
    jet_min = jet_var.min().item()
    jet_max = jet_var.max().item()

    jet_bins = []
    bins_true = []
    number_bins = len(bins) - 1
    # Assign each jet to a bin
    for i in range(len(jet_var)):
        for j in range(len(bins)-1):
            if j < len(bins) - 2 and jet_var[i] >= bins[j] and jet_var[i] < bins[j+1]:
                jet_bins.append(j)
                break
            elif j == len(bins) - 2 and jet_var[i] >= bins[j] and jet_var[i] <= bins[j+1]:
                jet_bins.append(j)

    for i in range(number_bins):
        bins_true.append([])

    for i in range(len(jet_bins)):
        bins_true[jet_bins[i]].append(jet_true[i].item())
    
    for i in range(number_bins):
        bins_true[i] = np.array(bins_true[i])

    # Prepare to plot x-axis
    for i in range(len(bins)-2, 0, -1):
        bins.insert(i, bins[i])


    # Get discriminator scores for each run + model
    for m, models_results in enumerate(jet_results):
        for r, run_results in enumerate(models_results):
            discriminator = get_discriminator(run_results)
            discriminator = discriminator[0]

            results = []
            for i in range(number_bins):
                results.append([])

            for i in range(len(jet_var)):
                results[jet_bins[i]].append(discriminator[i])
            
            
            jet_results[m][r] = []
            for i in range(number_bins):
                jet_results[m][r].append(results[i])


    working_points = [0.70, 0.85]

    for wp in working_points:
        min_, max_ = np.inf, -np.inf
        for flav in (0, 1, 2):
            fig, ax = plt.subplots(2, figsize=(8,8), gridspec_kw={'height_ratios': [2, 1]})
            models_acc = []
            for m, model_results in enumerate(jet_results):
                model_acc = []
                N_b = np.zeros(number_bins)
                m_b = np.zeros(number_bins)
                for r, results in enumerate(model_results): # each sample of the model
                    acc = []
                    for b in range(number_bins):
                        results[b] = np.array(results[b])
                        if flav != 0:
                            D = compute_threshold_eff(bins_true[b], results[b], wp)
                        else:
                            # D = compute_threshold_rej(bins_true[b], results[b])
                            D = compute_threshold_eff(np.concatenate(bins_true), np.concatenate(results), wp)
                        assert(results[b].shape == bins_true[b].shape)
                        # FIXME wrong arithmetic (but easily adaptable for rejection)
                        # res = results[b][bins_true[b] == 0]
                        # acc.append(len(res[res > D]) / len(res))
                        res = results[b][bins_true[b] == flav]
                        N_b[b] += len(res)
                        m_b[b] += len(res[res <= D])
                        if flav == 0:
                            acc.append(len(res[res > D]) / len(res))
                        else:
                            acc.append(len(res) / len(res[res > D]))
                    model_acc.append(np.array(acc))
                N_b = N_b / len(model_results)
                m_b = m_b / len(model_results)
                model_acc_mean_aux = np.mean(np.array(model_acc), axis=0)
                model_acc_std_aux = (1 / N_b) * np.sqrt(m_b * (1 - m_b / N_b))
                print("STD N m", model_acc_std_aux, N_b, m_b, "---", sep='\n')
                model_acc_mean = []
                model_acc_std = []
                for i in range(len(model_acc_mean_aux)):
                    model_acc_mean.append(model_acc_mean_aux[i])
                    model_acc_mean.append(model_acc_mean_aux[i])
                    model_acc_std.append(model_acc_std_aux[i])
                    model_acc_std.append(model_acc_std_aux[i])
                model_acc_mean = np.array(model_acc_mean)
                model_acc_std = np.array(model_acc_std)
                min_ = min(model_acc_mean.min(), min_)
                max_ = max(model_acc_mean.max(), max_)
                std_upper = model_acc_mean + model_acc_std
                std_lower = model_acc_mean - model_acc_std
                lbl = labels[m]
                for qq in range(0, len(bins)+1, 2):

                    ax[0].scatter(0,  -10, marker=markers[m], color=colors[m], label=lbl)
                    ax[0].plot(bins[qq:qq+2], model_acc_mean[qq:qq+2], color=colors[m], linewidth=1)
                    if qq < len(bins):
                        ax[0].scatter(bins[qq]+ 0.5*(bins[qq+1] - bins[qq]), model_acc_mean[qq],marker=markers[m], s=50, color=colors[m])
                    lbl = "_nolegend_"
                    ax[0].fill_between(bins[qq:qq+2], std_lower[qq:qq+2], std_upper[qq:qq+2], color=colors[m], alpha=0.2)

                models_acc.append(model_acc_mean)
            if flav == 0:
                ax[0].set_ylabel("b-jet efficiency ($\epsilon_b$)")
                ax[0].set_ylim(0.6, 1)
            elif flav == 1:
                ax[0].set_ylabel(r"c-jet rejection ($1/\epsilon_c$)")
                ax[0].set_ylim(min_, 1.05 * max_)
            elif flav == 2:
                ax[0].set_ylabel(r"l-jet rejection ($1/\epsilon_l$)")
                ax[0].set_ylim(min_, 1.05 * max_)
            
            ax[0].set_xlim(jet_min,jet_max)
            ax[0].legend(loc="lower right" if var != "eta" else "lower center", handlelength=1)

            # Ratio panel (denominator is always first model)
            baseline_acc = models_acc[0]    
            for r in range(len(jet_results)):
                model_acc = models_acc[r]
                ratio = model_acc / baseline_acc
                lbl = labels[r]
                for qq in range(0, len(bins)+1, 2):
                    ax[1].plot(bins[qq:qq+2], ratio[qq:qq+2], color=colors[r], linewidth=1)
                    if qq < len(bins):
                        ax[1].scatter(bins[qq]+ 0.5*(bins[qq+1] - bins[qq]), ratio[qq], marker=markers[r], s=50, color=colors[r])
                    lbl = "_nolegend_"

            ax[1].set_xlabel(var_label)
            ax[1].set_ylabel("Ratio to " + labels[0])
            ax[1].set_xlim(jet_min,jet_max)
            # ax[1].legend(loc="best")
            if flav > 0:
                flag = "c" if flav == 1 else "l"
                with open("configs.json", "r") as f:
                    settings = json.load(f)
                    ax[0].set_ylim(*settings[f'limits_bins_{100*wp}_{flag}'])
                    ax[1].set_ylim(*settings[f'limits_bins_{100*wp}_{flag}_ratio'])
            plt.tight_layout()


            plt.savefig(f"{save_dir}/ATLAS_eff_ensemble_flav{flav}_{var}_{wp}.pdf")

            plt.close()
    reset_font()
    return 


def compare_models_discriminator_ATLAS(jet_results, jet_true, save_dir, labels):
    font = {'family' : 'sans-serif',
        'size'   : 14}
    plt.rc('font', **font)
    discriminators_b = []
    discriminators_c = []
    discriminators_u = []

    for t in range(len(jet_results)):
        discriminators_model = []
        for inst in range(len(jet_results[t])):
            d_inst = get_discriminator(jet_results[t][inst])[0]
            discriminators_model.append(d_inst)
        discriminators_model = np.array(discriminators_model)
        discriminator_model = np.mean(discriminators_model, axis=0)
        jets_b = jet_true == 0
        jets_c = jet_true == 1
        jets_u = jet_true == 2
        discriminators_u.append(discriminator_model[jets_u])
        discriminators_c.append(discriminator_model[jets_c])
        discriminators_b.append(discriminator_model[jets_b])
    
    color_b = (46/255, 139/255, 86/255, 1.0) # "#2E8B56".lower() # 'forestgreen'
    color_c = (255/255, 215/255, 0, 1.0) # "#FFD700 ".lower() # 'gold'
    color_u = (100/255, 149/255, 237/255, 1.0) # "#6495ED".lower() # 'dodgerblue'
    for t in range(1, len(jet_results)):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        fc = 0.05
        bins = np.linspace(-8,10,250)

        ax.hist(discriminators_u[0], bins=bins, histtype='step', color=color_u, label='l-jets '+labels[0], linestyle='-', linewidth=5, density=True)
        ax.hist(discriminators_c[0], bins=bins, histtype='step', color=color_c, label='c-jets '+labels[0], linestyle='-', linewidth=5, density=True)
        ax.hist(discriminators_b[0], bins=bins, histtype='step', color=color_b, label='b-jets '+labels[0], linestyle='-', linewidth=5, density=True)

        ax.hist(discriminators_u[t], bins=bins, histtype='step', color=color_u, label='l-jets '+labels[t], linestyle=':', linewidth=5, density=True)
        ax.hist(discriminators_c[t], bins=bins, histtype='step', color=color_c, label='c-jets '+labels[t], linestyle=':', linewidth=5, density=True)
        ax.hist(discriminators_b[t], bins=bins, histtype='step', color=color_b, label='b-jets '+labels[t], linestyle=':', linewidth=5, density=True)

        ax.legend(loc="upper left")
        # ax.legend(bbox_to_anchor=(1.0, 1.03), loc="upper left")

        ax.set_xlabel('$D_b$')
        ax.set_ylabel('Arbitrary Units')
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/discriminator_{labels[0]}_vs_{labels[t]}.pdf")

    reset_font()


# High-level functions
def plot_classifications(predictions, targets, labels, labels_key):
    predictions = np.array(predictions)
    predictions_ = predictions
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
        except Exception as e:
            raise e
            print("Standard ROC not plotted.")



def performance_cmp_jets(predictions, true_flavours, labels, jet_pts, jet_etas, jet_ntracks):
    def equal_frequency(x, nbin):
        nlen = len(x)
        return list(np.interp(np.linspace(0, nlen, nbin + 1),
                        np.arange(nlen),
                        np.sort(x)))
    results = []
    for t in range(len(predictions)):
        model_results = []
        for inst in range(len(predictions[t])):
            # data: (efficiency, c-rejection, u-rejection)
            data = prepare_ROC_jet_ATLAS_bcujets(predictions[t][inst], true_flavours)
            model_results.append(list(data))
            
        results.append(model_results)

    with open("configs.json", "r") as f:
        settings = json.load(f)
        results_dir = settings['results_dir']
        colors = settings['colors']
        linestyles = settings['linestyles']
        markers = settings['markers']
        for c in range(len(colors)):
            if colors[c][0] == "(":
                colors[c] = eval(colors[c])
        for l in range(len(linestyles)):
            if linestyles[l][0] == "(":
                linestyles[l] = eval(linestyles[l])

    compare_models_discriminator_ATLAS(predictions, true_flavours, results_dir, labels)
    compare_models_jet_ATLAS(results, colors, linestyles, markers, results_dir, labels)
    compare_models_eff_ATLAS(copy.deepcopy(predictions), colors, linestyles, markers, results_dir, labels, jet_pts, true_flavours, equal_frequency(jet_pts, 5), "pt", r"Jet $p_{T}$ [GeV]")
    compare_models_eff_ATLAS(copy.deepcopy(predictions), colors, linestyles, markers, results_dir, labels, jet_etas, true_flavours, equal_frequency(jet_etas, 5), "eta", r"Jet $\eta$")
    compare_models_eff_ATLAS(copy.deepcopy(predictions), colors, linestyles, markers, results_dir, labels, jet_ntracks, true_flavours, equal_frequency(jet_ntracks, 5), "n_tracks", r"#Tracks")
    # compare_models_eff_ATLAS(copy.deepcopy(predictions), colors, results_dir, labels, jet_pts, true_flavours, [20, 40, 60, 80, 100, 200], "pt", r"Jet $p_{T}$ [GeV]")
    # compare_models_eff_ATLAS(copy.deepcopy(predictions), colors, results_dir, labels, jet_etas, true_flavours, [-2.5, -1, -.5, 0, .5, 1, 2.5], "eta", r"Jet $\eta$")
    # compare_models_eff_ATLAS(copy.deepcopy(predictions), colors, results_dir, labels, jet_ntracks, true_flavours, [1, 4, 8, 10, 12, 15], "n_tracks", r"#Tracks")
