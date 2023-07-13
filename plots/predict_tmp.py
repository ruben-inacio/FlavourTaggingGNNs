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
from plot_utils import make1Dplots_3D

# FIXME update
def evaluate_track_compatibility(model, dl): 

    for origin in range(4):
        pred_vtx, true_vtx = [], []
        for i, d in enumerate(dl):
            if torch.cuda.is_available(): 
                d = d.cuda()
            
            if d.x.shape[1] > 8:
                d.x = d.x[:, :-7]
            try:
                _, _, out_vtx, _, _, _, mask_edges = model(d.x, d.edge_index, d.batch)
            except Exception:
                try:
                    _, _, out_vtx, _, _, mask_edges = model(d.x, d.edge_index, d.batch)
                except Exception:
                    _, _, out_vtx, _, mask_edges = model(d.x, d.edge_index, d.batch)

            mask_origin_nodes, mask_origin_edges = filter_tracks(d.trk_y, model.n_tracks, d.jet_y.shape[0], origin)

            mask_edges = mask_edges & mask_origin_edges

            repeated = torch.tril(torch.arange(0, model.n_tracks**2).view(-1, model.n_tracks), -1).view(-1).bool()
            repeated = repeated.expand(d.x.shape[0] // model.n_tracks, repeated.shape[0]).reshape(-1)
            repeated = repeated.to(mask_edges.device)
            mask_edges = mask_edges & repeated

            mask_edges = mask_edges.clone().detach().requires_grad_(False)
            out_vtx = out_vtx.clone().detach().requires_grad_(False)
            d.edge_y = d.edge_y.clone().detach().requires_grad_(False)
           
            pred_vtx.append(out_vtx[mask_edges])
            true_vtx.append(d.edge_y[mask_edges])

            break

        pred_vtx = torch.cat(pred_vtx)
        true_vtx = torch.cat(true_vtx)
            
        true_vtx = true_vtx.cpu().detach().numpy()
        pred_vtx = pred_vtx.cpu().detach().numpy()

        evaluate_classification(true_vtx, pred_vtx, ['Distinct', 'Common'], 'test'+str(origin), save_plot=True, verbose=True)




def performance_classification(model, dl):
    print("Evaluating classification")
    try:
        evaluate_track_compatibility(model, dl)
        evaluate_performance(model, dl)
    except Exception as e:
        print("Evaluating classification: FAILED")
        print(e)
        raise e


def get_info_reg(dl, scy=None):
    true = []
    true_flavours = []
    jet_pts = []
    jet_trks = []
    dl.pin_memory_device = []
    for i, dd in enumerate(dl):
        for j in range(dd.x.shape[0] // N_JETS):
            x = dd.x[N_JETS*j:N_JETS*(j+1), :, :]
            y = dd.y[N_JETS*j:N_JETS*(j+1), :, :]

            x = np.array(x)
            y = np.array(y)
            m = np.any(np.array(x)[:,:, :16] != 1., axis=2)
            trks = m.sum(axis=1)

            pts = x[:, :, 0]
            pts = np.where(m, pts, 0)
            pts = np.sum(pts, axis=1)
            jet_pts.append(pts)
            jet_trks.append(trks)
        
            batch = {}
            batch['jet_y'] = y[:, 0, 18:21]

            batch['trk_y'] = y[:, :, 21:25]
            batch['edge_y'] = y[:, :, 3:18]
            batch['trk_vtx'] = x[:, 16:19]
            batch['jet_vtx'] = x[:, 0, 19:22] # [::15]
            
            this_true = batch['jet_vtx'] # d.jet_vtx.reshape(-1, 3)
            if scy is not None:
                this_true = np.array(scy.inverse_transform(this_true))
            true.append(this_true)

            true_flavours.append(np.argmax(batch['jet_y'], axis=1))
        
    # true = np.concatenate(true)
    # true_flavours = np.concatenate(true_flavours)
    # jet_pts = np.concatenate(jet_pts)
    # jet_trks = np.concatenate(jet_trks)
    true = np.array(true).reshape(-1, 3)
    true_flavours = np.array(true_flavours).reshape(-1)
    jet_pts = np.array(jet_pts).reshape(-1)
    jet_trks = np.array(jet_trks).reshape(-1)
    return true, true_flavours, jet_pts, jet_trks   

# predictions, targets, name, label=None, color=None
  

euc_dist = lambda x, y: np.sqrt(np.sum((x - y)**2, axis=1))
difference = lambda x, y: np.sum((x - y), axis=1)
difference_x = lambda x, y: x[:, 0] - y[:, 0]
difference_y = lambda x, y: x[:, 1] - y[:, 1]
difference_z = lambda x, y: x[:, 2] - y[:, 2]
euc_dist_norm = lambda x, y: .5 * np.var(x - y, axis=1) / (np.var(x, axis=1) + np.var(y, axis=1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-results_dir')
    parser.add_argument('-input_suffix', default=DEFAULT_SUFFIX)
    parser.add_argument('-model_dir', default="../models")
    parser.add_argument('-name', default="checkpoint_tmp_combined") #"baseline_all_tasksv2_sep")
    parser.add_argument('-mclass', default=MODEL_CLASS)
    parser.add_argument('-ensemble_size', type=int, default=1)
    # parser.add_argument('-task', choices=["node", "node_jet", "regression"], default="node")
    return parser.parse_args()    



# FIXME only works for vtx fitting
if __name__ == "__main__":
    opt = parse_args()
    

    IMG_DIR = "../reg_losses/"

    if not os.path.exists(IMG_DIR): 
        os.makedirs(IMG_DIR)
    
    # test_dl = torch.load("%s/test_dl.pth"%(opt.input_dir))
    # with open("../models/regression_compute_baseline/params_0.pickle", 'rb') as fp:
    #     params = pickle.load(fp)
    # model = Predictor(
    #     hidden_channels=    32,
    #     heads=              2,
    #     layers=             3,
    #     strategy_sampling=  "compute",
    #     method=             "regression"
    # )
    #############
    # model = NDIVE()
    # params = checkpoints.restore_checkpoint(ckpt_dir='rachel', target=None, step=0, parallel=False)['params']
    #############
    
    # preds_instance, _, targets, flavours, jet_pts, jet_trks = get_predictions_reg(model, params, test_dl)
    
    targets  = np.load('../ground_truth/jet_vtx.npy')
    flavours = np.load('../ground_truth/jet_y.npy')
    jet_pts  = np.load('../ground_truth/jet_pts.npy')
    jet_trks = np.load('../ground_truth/jet_trks.npy')
    true_nodes = np.load('../ground_truth/trk_y.npy')
    true_edges = np.load('../ground_truth/edge_y.npy')

    # targets, flavours, jet_pts, jet_trks = get_info_reg(test_dl)

    models = ['models2/reg_ste', 'models/regression_02mse'] #', 'models/gn2_idgnns']
    # models = ['models/tn1fitrachel2']
    labels = ['ori', "02mse",]
    # preds_instance = np.load(f"../models/rachel/results_graph_reg_var_0.npy")
    # print(preds_instance.tolist())
    # exit(0) xprev-xtrue 
    # SHAPE: |models| x |ensemble| x num_samples x num_classes
    predictions = []  # Regression (num_classes = 3)
    predictions_graph_clf = []  # Jet ROC Curve (num_classes = 3)
    predictions_nodes_clf = []
    predictions_edges_clf = []

    do_regression = True
    do_jet_roc = True
    do_classifications = True
    for m in models:
        predictions.append([])
        predictions_graph_clf.append([])
        predictions_nodes_clf.append([])
        predictions_edges_clf.append([])


        for run in range(opt.ensemble_size):
            try:
                preds_instance = np.load(f"../{m}/results_graph_reg_{run}.npy")
                assert(preds_instance.ndim == 2 and preds_instance.shape[1] == 3)
                assert(preds_instance.shape[0] == jet_pts.shape[0])
                assert(preds_instance.shape[0] == flavours.shape[0])
                assert(preds_instance.shape[0] == jet_trks.shape[0])
                predictions[-1].append(preds_instance)
            except Exception:
                print("Not doing regression")
                do_regression = False

            try:
                preds_graph_clf_instance = np.load(f"../{m}/results_graph_clf_{run}.npy")
                assert(preds_graph_clf_instance.shape[0] == flavours.shape[0])
                predictions_graph_clf[-1].append(preds_graph_clf_instance)

                preds_nodes_clf_instance = np.load(f"../{m}/results_nodes_clf_{run}.npy")
                assert(preds_nodes_clf_instance.shape[0] == true_nodes.shape[0])
                predictions_nodes_clf[-1].append(preds_nodes_clf_instance)

                preds_edges_clf_instance = np.load(f"../{m}/results_edges_clf_{run}.npy")
                assert(preds_edges_clf_instance.shape[0] == true_edges.shape[0])
                predictions_edges_clf[-1].append(preds_edges_clf_instance)

            except Exception as e:
                print("Not doing jet ROC, classifications")
                print(e)
                do_jet_roc = False
                do_classifications = False

            errors = None

    if do_regression:
        assert(len(predictions) == len(models))
        assert(all(len(predictions[x]) == opt.ensemble_size for x in range(opt.ensemble_size)))
        performance_regression(labels, predictions, errors, targets, flavours, jet_pts, jet_trks)
    if do_jet_roc:
        performance_roc_jets(predictions_graph_clf, flavours, labels)
    if do_classifications:
        performance_cm_classifications(
            predictions_graph_clf, 
            predictions_nodes_clf, 
            predictions_edges_clf,
            flavours,
            true_nodes,
            true_edges,
            labels
        )
