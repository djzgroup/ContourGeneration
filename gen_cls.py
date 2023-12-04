import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.dataset_cls import ShapeNetCore_cls
from utils.misc import *
from utils.data import *
from models.autoencoder import *
from evaluation import EMD_CD


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt_0.009121_400000_chair.pt')
parser.add_argument('--categories', type=str_list, default=['chair'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/chair_gen_cls.hdf5')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_points', type=int, default=1024)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_Ours_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
# test_dset = ShapeNetCore(
#     path=args.dataset_path,
#     cates=args.categories,
#     split='train',
#     scale_mode=ckpt['args'].scale_mode
# )
# test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)


test_raw_dset = ShapeNetCore_cls(
    path=args.dataset_path,
    cates=args.categories,
    split='raw_train',
    scale_mode=ckpt['args'].scale_mode
)
test_raw_loader = DataLoader(test_raw_dset, batch_size=args.batch_size, num_workers=0)
# Model
logger.info('Loading model...')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

# all_ref = []
all_recons = []
all_com=[]
for i, batch in enumerate(tqdm(test_raw_loader)):
    raw = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)
    model.eval()
    with torch.no_grad():
        code = model.encode(raw)
        recons = model.decode(code, args.num_points, flexibility=ckpt['args'].flexibility).detach()

    recons = recons * scale + shift

    all_recons.append(recons.detach().cpu())
    all_com.append(raw.detach().cpu())
all_recons = torch.cat(all_recons, dim=0)
all_com=torch.cat(all_com,dim=0)


# for i in range (0,19):
#     # all_recons = all_recons.to('cpu')
#     # all_recons = np.array(all_recons)
#     recons=all_recons[i][:][:]
#     np.savetxt('./gen_cls/17/%d.txt' % i, recons)
#
# for i,batch in enumerate(tqdm(test_loader)):
#     ref=batch['pointcloud'].to(args.device)
#     shift = batch['shift'].to(args.device)
#     scale = batch['scale'].to(args.device)
#     model.eval()
#     ref = ref * scale + shift
#     all_ref.append(ref.detach().cpu())
# all_ref = torch.cat(all_ref, dim=0)

logger.info('Saving point clouds...')
# np.save(os.path.join(save_dir, 'ref_chair_.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out_chair_cls.npy'), all_recons.numpy())
np.save(os.path.join(save_dir, 'com_chair_cls.npy'), all_com.numpy())
#
# logger.info('Start computing metrics...')
# metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
# cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
# logger.info('CD:  %.12f' % cd)
# logger.info('EMD: %.12f' % emd)
