import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from models.HisToGene_model import HisToGene
from models.STNet_model import STModel
from models.UNI import UNI
from models.WSUNI import WSUNI
from models.WS_UNI import WS_UNI
from utils import *
from predict import model_predict, uni_predict, wsuni_predict, wsuni_slicelevel_predict, get_R, cluster, get_MSE, get_MAE
from dataset import ViT_HER2ST, HER2ST, UNI_HER2ST, WSUNI_HER2ST, WS_UNI_HER2ST, WSUNI_SliceLevel_HER2ST


fold = 5
tag = '-htg_her2st_785_32_cv'

#normal histogene prediction

mode = 'WSUNI_2'# input("Choose model to predict [Histogene/ST-Net]: ")
if mode == "Histogene":
    model = HisToGene.load_from_checkpoint("model_ckpts/histogene_last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = ViT_HER2ST(train=False,sr=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)

    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)



    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='histogene_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='histogene_FASN.png')

elif mode == "UNI":
    model = UNI.load_from_checkpoint("model_ckpts/UNI_final/UNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=39.ckpt", n_genes=785, learning_rate=1e-5, max_epochs=50)
    device = torch.device("cuda")
    dataset = UNI_HER2ST(train=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = uni_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)

    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='UNI_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='UNI_FASN.png')
    
elif mode == "WSUNI":
    model = WSUNI.load_from_checkpoint("model_ckpts/WSUNI_adapk_50epoch/WSUNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=44.ckpt", n_genes=785, learning_rate=1e-5, max_epochs=50)
    device = torch.device("cuda")
    dataset = WSUNI_HER2ST(train=False, fold=fold, cache_dir='cache_features_test_saved', topk=40)
    test_loader = DataLoader(dataset, batch_size=16, num_workers=1)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = wsuni_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)

    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='WSUNI_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='WSUNI_FASN.png')
    
if mode == "WSUNI_2":
    model = WSUNI.load_from_checkpoint("model_ckpts/WSUNI_adapk_50epoch/WSUNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=49.ckpt", n_genes=785, learning_rate=1e-5, max_epochs=50)
    device = torch.device("cuda")
    dataset = WSUNI_SliceLevel_HER2ST(train=False, fold=fold, cache_dir='cache_features_test_saved', topk=40)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)
   
    label = None
    print(len(dataset))
    #iterate over labels of test set
    for i in range(len(dataset)):
        if label is None:
            label=dataset.label[dataset.names[i]]
            # print(label.shape)
        else:
            temp=dataset.label[dataset.names[i]]
            label=np.concatenate((label,temp))
        # print(temp.shape)
    # print(label)
    # print(label.shape)
    # print(dataset.names)
    print("check bef")
    adata_pred, adata_gt, patches_count = wsuni_slicelevel_predict(model, test_loader, attention=False, device = device)
    print("check")
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred.shape)
    print(adata_gt.shape)
    print(adata_pred, adata_gt)
    print(patches_count)
    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    def post_process_prediction(pred, ratio = 0.005, gamma=2.0):
        n_total = pred.numel()
        k = max(1, int(n_total * ratio))

        # Flatten, lấy top-k ngưỡng
        flat = pred.flatten()
        threshold_val = torch.topk(flat, k).values[-1]

        # Apply top-k mask
        pred_masked = torch.where(pred >= threshold_val, pred, torch.tensor(0.0, device=pred.device))

        # Optional normalize per gene (column) after masking
        pred_min = pred_masked.min(dim=0, keepdim=True)[0]
        pred_max = pred_masked.max(dim=0, keepdim=True)[0]
        pred_norm = (pred_masked - pred_min) / (pred_max - pred_min + 1e-8)

        # Optional sharpen
        pred_sharp = pred_norm ** gamma

        return pred_sharp

    pred_np = adata_pred.X  # shape (n_spots, n_genes)
    pred = torch.tensor(pred_np, dtype=torch.float32)
    pred_pp = post_process_prediction(pred, ratio=1.0, gamma=2.0)
    #adata_pred.X = pred_pp.numpy()
    clus, ARI = cluster(adata_pred, label)

    print('ARI:',ARI)
    
    MSE = get_MSE(adata_pred, adata_gt)
    MAE = get_MAE(adata_pred, adata_gt)
    print('MSE:', np.nanmean(MSE))
    print('MAE:', np.nanmean(MAE))

    import matplotlib.image as mpimg
    img = mpimg.imread("/home/jovyan/shared/tienhuu060102/spatial-transcriptomics/E1_new.png")

    trace = 0
    for i in range(len(dataset)):
        print("patch_len:", patches_count[i])
        patch = adata_pred[trace:trace+patches_count[i]]
        gt = adata_gt[trace:trace+patches_count[i]]
        trace += patches_count[i]
        # print(patch)

        #visualize results
        sc.pl.spatial(adata_pred, img=img, color='kmeans', spot_size=112, frameon=False,
        legend_loc=None,title=None,
        show=False)

        ax = plt.gca()
        ax.set_title("")  # Remove title

        os.makedirs('figures/kmeans', exist_ok=True) 
        os.makedirs('figures/FASN', exist_ok=True) 

        plt.savefig(f"figures/kmeans/WSUNI_kmeans_{dataset.names[i]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.clf()
        plt.close()

        sc.pl.spatial(adata_pred, img=img, color='FASN', spot_size=112, frameon=False,
        legend_loc=None, title=None,
        show=False,color_map='magma')

        ax = plt.gca()
        ax.set_title("")  # Remove title

        plt.savefig(f"figures/FASN/WSUNI_FASN_{dataset.names[i]}_new.png", dpi=300, bbox_inches='tight', transparent=True)
        plt.clf()
        plt.close()
    
if mode == "WS_UNI":
    model = WS_UNI.load_from_checkpoint("model_ckpts/WS_UNI_v2/WS_UNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=29.ckpt", n_layers=6, uni_ckpt_path="model_ckpts/UNI_v2/UNI_every5epoch_-htg_her2st_785_32_cv_5_epoch=44.ckpt", n_genes=785, learning_rate=1e-5, max_epochs=50)
    device = torch.device("cuda")
    dataset = WS_UNI_HER2ST(train=False,sr=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = wsuni_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)

    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)



    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='WS_UNI_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='WS_UNI_FASN.png')

elif mode == "ST-Net":
    model = STModel.load_from_checkpoint("model_ckpts/stnet_last_train_"+tag+'_'+str(fold)+".ckpt", n_genes=785, learning_rate=1e-5)
    device = torch.device("cuda")
    dataset = HER2ST(train=False,fold=fold)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    label=dataset.label[dataset.names[0]]
    adata_pred, adata_gt = model_predict(model, test_loader, attention=False, device = device)
    adata_pred = comp_tsne_km(adata_pred,4)

    g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
    adata_pred.var_names = g
    sc.pp.scale(adata_pred)
    # print(adata_pred)
    print(adata_pred, adata_gt)


    R=get_R(adata_pred,adata_gt)[0]
    print('Pearson Correlation:',np.nanmean(R))
    clus,ARI=cluster(adata_pred, label)
    print('ARI:',ARI)

    #visualize results
    sc.pl.spatial(adata_pred, img=None, color='kmeans', spot_size=112, save='ST-Net_kmeans.png')

    sc.pl.spatial(adata_pred, img=None, color='FASN', spot_size=112, color_map='magma', save='ST-Net_FASN.png')