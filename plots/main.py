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
# from plot_utils import make1Dplots_3D
from plot_classifications import *
from plot_vertex_fitting import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir')
    parser.add_argument('-ensemble_size', type=int, default=1)
    parser.add_argument('-regression', type=bool, default=False)
    parser.add_argument('-atlas_roc', type=bool, default=True)
    parser.add_argument('-classifications', type=bool, default=True)
    return parser.parse_args()
   

if __name__ == "__main__":
    opt = parse_args()
    do_regression = opt.regression
    do_jet_cmp = opt.atlas_roc
    do_classifications = opt.classifications
    
    with open("configs.json", "r") as f:
        settings = json.load(f)


    if not os.path.exists(settings['results_dir']): 
        os.makedirs(settings['results_dir'])
        os.makedirs(settings['results_dir'] + "/pt")
        os.makedirs(settings['results_dir'] + "/ntrks")

    models = settings['models']

    # Ground truth information
    jet_pts    = np.load(f'{settings["ground_truth"]}/jet_pts.npy')
    jet_etas   = np.load(f'{settings["ground_truth"]}/jet_etas.npy')
    jet_trks   = np.load(f'{settings["ground_truth"]}/jet_trks.npy')
    true_vtx   = np.load(f'{settings["ground_truth"]}/jet_vtx.npy')
    true_nodes = np.load(f'{settings["ground_truth"]}/trk_y.npy')
    true_edges = np.load(f'{settings["ground_truth"]}/edge_y.npy')
    true_graph = np.load(f'{settings["ground_truth"]}/jet_y.npy')
    assert(jet_pts.shape[0] == jet_trks.shape[0])
    assert(jet_pts.shape[0] == true_graph.shape[0])
    true_nodes = true_nodes.reshape(-1)
    valid_nodes = true_nodes > -1 
    true_nodes = true_nodes[valid_nodes]

    true_edges = true_edges.reshape(-1,)
    valid_edges = true_edges > -1 
    true_edges = true_edges[valid_edges]

    # print(jet_pts.shape, jet_trks.shape, true_vtx.shape, true_nodes.shape, true_edges.shape, true_graph.shape)
    # Obtain predictions for each model
    predictions_fitting   = []
    predictions_errors    = []
    predictions_graph_clf = []
    predictions_nodes_clf = []
    predictions_edges_clf = []

    for m in models:
        predictions_fitting.append([])
        predictions_errors.append([])
        predictions_graph_clf.append([])
        predictions_nodes_clf.append([])
        predictions_edges_clf.append([])

        for run in range(opt.ensemble_size):
            try: 
                results_model_run = np.load(f"{m}/results_{run}.npz")
                old_files = False
            except Exception:
                old_files = True

            try:
                if old_files:
                    results_fitting = np.load(f"{m}/results_graph_reg_{run}.npy")
                else:
                    results_fitting = results_model_run['results_graph_reg']

                assert(results_fitting.ndim == 2 and results_fitting.shape[1] == 3)
                assert(results_fitting.shape[0] == true_graph.shape[0])

                if old_files:
                    results_errors = np.load(f"{m}/results_graph_reg_var_{run}.npy")
                else:
                    results_errors = results_model_run['results_graph_reg_var']

                if results_errors.ndim == 3:
                    assert(results_errors.shape[1] == results_errors.shape[2] == 3)
                    diag = [
                        results_errors[:, 0, 0].reshape(-1, 1),
                        results_errors[:, 1, 1].reshape(-1, 1),
                        results_errors[:, 2, 2].reshape(-1, 1)
                    ]
                    results_errors = np.concatenate(diag, axis=1)
                assert(results_errors.ndim == 2 and results_errors.shape[1] == 3)
                assert(results_errors.shape[0] == true_graph.shape[0])

                predictions_fitting[-1].append(results_fitting)
                predictions_errors[-1].append(results_errors)
            except Exception:
                print("Fitting plots disabled.")
                do_regression = False

            try:
                if old_files:
                    results_graph_clf = np.load(f"{m}/results_graph_clf_{run}.npy")
                else:
                    results_graph_clf = results_model_run['results_graph_clf']
                if results_graph_clf.ndim == 4:
                    results_graph_clf = results_graph_clf.reshape(-1, 3)
                assert(results_graph_clf.shape[0] == true_graph.shape[0])
                predictions_graph_clf[-1].append(results_graph_clf)

            except Exception as e:
                print("ATLAS ROC, classification plots disabled.")
                do_jet_cmp = False
                do_classifications = False

                # raise e
            
            try:
                assert(do_classifications)
                if old_files:
                    results_nodes_clf = np.load(f"{m}/results_nodes_clf_{run}.npy")
                else:
                    results_nodes_clf = results_model_run['results_nodes_clf']

                results_nodes_clf = results_nodes_clf.reshape(-1, 4)
                if true_nodes.shape[0] < results_nodes_clf.shape[0]:
                    results_nodes_clf = results_nodes_clf[valid_nodes]
                assert(results_nodes_clf.shape[0] == true_nodes.shape[0])

                if old_files:
                    results_edges_clf = np.load(f"{m}/results_edges_clf_{run}.npy")
                else:
                    results_edges_clf = results_model_run['results_edges_clf']

                if results_edges_clf.ndim == 5:
                    results_edges_clf = results_edges_clf.reshape(-1, results_edges_clf.shape[-2], results_edges_clf.shape[-1])
                if results_edges_clf.ndim == 3:
                    results_edges_clf = results_edges_clf.reshape(-1, results_edges_clf.shape[-1])
                    if true_edges.shape[0] < results_edges_clf.shape[0]:
                        results_edges_clf = results_edges_clf[valid_edges]
                    if results_edges_clf.shape[-1] == 1:
                        results_edges_clf = np.concatenate([1 - results_edges_clf, results_edges_clf], axis=1)

                assert(results_edges_clf.shape[0] == true_edges.shape[0])

                predictions_nodes_clf[-1].append(results_nodes_clf)
                predictions_edges_clf[-1].append(results_edges_clf)
            except Exception as e:
                print("Classification plots disabled.")
                do_classifications = False
                # raise e

    labels = settings['labels_models']
    if do_regression:
        # Duplicate for jet_trks
        plot_discriminated_by_flavour(
            predictions_fitting, predictions_errors, true_vtx, true_graph, jet_pts, r"Jet $p_{T}$ [GeV]", labels, 'pt',[20, 40, 60, 80, 100, 200])
        plot_global_performance(
            predictions_fitting, predictions_errors, true_vtx, true_graph, jet_pts, r"Jet $p_{T}$ [GeV]", labels, 'pt', [20, 40, 60, 80, 100, 200])
        plot_discriminated_by_flavour(
            predictions_fitting, predictions_errors, true_vtx, true_graph, jet_trks, r"#Tracks", labels, 'ntrks', list(range(1, 16)))
        plot_global_performance(
            predictions_fitting, predictions_errors, true_vtx, true_graph, jet_trks, r"#Tracks", labels, 'ntrks', list(range(1, 16)))
        plot_fitting_average(predictions_fitting, predictions_errors, true_vtx, true_graph, jet_pts, labels, [20, 40, 60, 80, 100, 200])

    if do_jet_cmp:
        performance_cmp_jets(predictions_graph_clf, true_graph, labels, jet_pts, jet_etas, jet_trks)
        #performance_cmp_jets(predictions_graph_clf, true_graph, labels, jet_pts, jet_etas)

    if do_classifications:
        plot_classifications(predictions_graph_clf, true_graph, labels, "graph")
        plot_classifications(predictions_nodes_clf, true_nodes, labels, "nodes")
        plot_classifications(predictions_edges_clf, true_edges, labels, "edges")

