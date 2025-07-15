import torch
from torch.utils.data import DataLoader
from utils import *
from models.HisToGene_model import HisToGene
import warnings
from dataset import ViT_HER2ST
from tqdm import tqdm
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score as ari_score

MODEL_PATH = ''


def model_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    count = 0
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch, position)
            print(pred.shape, center.shape, exp.shape, sep = "\n")
            if preds is None:
                preds = pred #previously preds = pred.squeeze(); remove for compatibility w stnet
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)



    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt


def uni_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    count = 0
    with torch.no_grad():
        for patch_0, patch_1, patch_2, position, exp, center in tqdm(test_loader):

            patch_0, patch_1, patch_2 = patch_0.to(device), patch_1.to(device), patch_2.to(device)
            
            pred, cls_smallest, sim_loss = model(patch_0, patch_1, patch_2)

            if preds is None:
                preds = pred #previously preds = pred.squeeze(); remove for compatibility w stnet
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)



    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt


def wsuni_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    count = 0
    with torch.no_grad():
        for out_uni, cls_uni, cls_neighbors, position, exp, center in tqdm(test_loader):

            out_uni, cls_uni, cls_neighbors = out_uni.to(device), cls_uni.to(device), cls_neighbors.to(device)
            
            pred, cls, sim_loss = model(out_uni, cls_uni, cls_neighbors)

            if preds is None:
                preds = pred #previously preds = pred.squeeze(); remove for compatibility w stnet
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)



    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt

def wsuni_slicelevel_predict(model, test_loader, adata=None, attention=True, device=torch.device('cpu')):
    model.eval()
    model = model.to(device)
    patches_count = []
    preds_all = []
    centers_all = []
    gts_all = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            out_uni, cls_uni, cls_neighbors, cls_neighbors_loc, position, exp, center = batch

            # Flatten B x P --> (B*P)
            B, P, D = out_uni.shape
            out_uni = out_uni.view(B * P, -1).to(device)
            cls_uni = cls_uni.view(B * P, -1).to(device)
            cls_neighbors = cls_neighbors.view(B * P, cls_neighbors.shape[2], cls_neighbors.shape[3]).to(device)
            cls_neighbors_loc = cls_neighbors_loc.view(B * P, cls_neighbors_loc.shape[2], cls_neighbors_loc.shape[3]).to(device)
            center = center.view(B * P, -1).to(device)

            pred, cls, sim_loss = model(out_uni, cls_uni, center, cls_neighbors, cls_neighbors_loc)
            patches_count.append(pred.shape[0])
            preds_all.append(pred.cpu())
            centers_all.append(center.cpu())
            gts_all.append(exp.view(B * P, -1))

    preds = torch.cat(preds_all, dim=0).squeeze().numpy()
    ct = torch.cat(centers_all, dim=0).squeeze().numpy()
    gt = torch.cat(gts_all, dim=0).squeeze().numpy()

    # for i in range(1):
    #     print(preds[i], gt[i], sep = "\n")
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt, patches_count

def ws_uni_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    count = 0
    with torch.no_grad():
        for patches_0, patches_1, patches_2, position, exp, center in tqdm(test_loader):

            patches_0, patches_1, patches_2, position = patches_0.to(device), patches_1.to(device), patches_2.to(device), position.to(device)
            
            pred, _ = model(patches_0, patches_1, patches_2, position)

            if preds is None:
                preds = pred #previously preds = pred.squeeze(); remove for compatibility w stnet
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)



    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct

    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct

    return adata, adata_gt

def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1


def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)

from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_MSE(data1, data2, dim=1):
    adata1 = data1.X
    adata2 = data2.X
    mse_list = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            mse = mean_squared_error(adata1[:, g], adata2[:, g])
        elif dim == 0:
            mse = mean_squared_error(adata1[g, :], adata2[g, :])
        mse_list.append(mse)
    return np.array(mse_list)

def get_MAE(data1, data2, dim=1):
    adata1 = data1.X
    adata2 = data2.X
    mae_list = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            mae = mean_absolute_error(adata1[:, g], adata2[:, g])
        elif dim == 0:
            mae = mean_absolute_error(adata1[g, :], adata2[g, :])
        mae_list.append(mae)
    return np.array(mae_list)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for fold in [5,11,17,26]:
    for fold in range(12):
        # fold=30
        # tag = '-vit_skin_aug'
        # tag = '-cnn_her2st_785_32_cv'
        tag = '-vit_her2st_785_32_cv'
        # tag = '-cnn_skin_134_cv'
        # tag = '-vit_skin_134_cv'
        ds = 'HER2'
        # ds = 'Skin'

        print('Loading model ...')
        # model = STModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = VitModel.load_from_checkpoint('model/last_train_'+tag+'.ckpt')
        # model = STModel.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt") 
        model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt")
        model = model.to(device)
        # model = torch.nn.DataParallel(model)
        print('Loading data ...')

        # g = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        g = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        # dataset = SKIN(train=False,ds=ds,fold=fold)
        dataset = ViT_HER2ST(train=False,mt=False,sr=True,fold=fold)
        # dataset = ViT_SKIN(train=False,mt=False,sr=False,fold=fold)
        # dataset = VitDataset(diameter=112,sr=True)

        test_loader = DataLoader(dataset, batch_size=16, num_workers=4)
        print('Making prediction ...')

        adata_pred, adata = model_predict(model, test_loader, attention=False)
        # adata_pred = sr_predict(model,test_loader,attention=True)

        adata_pred.var_names = g
        print('Saving files ...')
        adata_pred = comp_tsne_km(adata_pred,4)
        # adata_pred = comp_umap(adata_pred)
        print(fold)
        print(adata_pred)

        adata_pred.write('processed/test_pred_'+ds+'_'+str(fold)+tag+'.h5ad')
        # adata_pred.write('processed/test_pred_sr_'+ds+'_'+str(fold)+tag+'.h5ad')

        # quit()

if __name__ == '__main__':
    main()

