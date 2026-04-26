from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import os

from torch.backends import cudnn
import torch.nn as nn
import random
from config import cfg
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model, TransNet_adaptive
from reid.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res

import warnings
warnings.filterwarnings("ignore")

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed) 
        random.seed(args.seed)
        
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False


    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
    elif 2 == args.setting:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']

    testing_only_set = [x for x in all_set if x not in training_set]
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)
    model_trans=TransNet_adaptive()
    model_trans2=TransNet_adaptive()

    model.cuda()
    model_trans.cuda()
    model_trans2.cuda()
    
    model = DataParallel(model)    
    model_trans = DataParallel(model_trans)
    model_trans2 = DataParallel(model_trans2)

    writer = SummaryWriter(log_dir=args.logs_dir)
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))
        copy_state_dict(checkpoint['state_dict'], model)
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)        
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

                         
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, step + 1)
            model = linear_combination(args, model, model_old, best_alpha)

            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1, logger_res=logger_res, feats_dir=args.logs_dir)

        exit(0)
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")

    # LSTKC++ long-term / short-term model state
    model_long = None   # Theta_t^l: long-term model (multi-domain balanced knowledge)
    model_short_old = None  # Theta_{t-1}^s: previous step's short-term trained model

    for set_index in range(0, len(training_set)):
        model_old = copy.deepcopy(model)

        # Determine old_model_long for C-STKR in trainer
        # t=1 (set_index=0): no old models
        # t=2 (set_index=1): single old model (STKR), no long-term yet
        # t>=3 (set_index>=2): both short-term and long-term old models (C-STKR)
        old_model_long_for_train = model_long if set_index >= 2 else None

        model, model_trans, model_trans2 = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, model_trans, model_trans2, out_channel,
                                            writer, old_model_long=old_model_long_for_train, logger_res=logger_res)

        # --- LSTKC++ post-training model management ---
        if set_index == 0:
            # t=1: no fusion, model_long not yet defined
            model_long = copy.deepcopy(model)
            model_short_old = copy.deepcopy(model)
        elif set_index == 1:
            # t=2: M*_e with Theta_2^l = Theta_1^s (= model_long from step 0)
            # model_long already holds Theta_1^s from previous iteration
            best_alpha = get_adaptive_alpha(args, model, model_long, all_train_sets, set_index)
            print('M*_e fusion weight delta={:.4f}'.format(best_alpha))
            model = linear_combination(args, model, model_long, best_alpha)
            model_short_old = copy.deepcopy(model_old)  # Theta_1^s for next step
            model_long = copy.deepcopy(model)            # Theta_2^l = fused model
        else:
            # t>=3: full LSTKC++ pipeline
            # M*_o: backward rectification to update long-term model
            dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[set_index]
            model_long = backward_rectification(args, model_short_old, model_long, init_loader)

            # M*_e: Eq.(15) delta, fuse Theta_t^s and Theta_t^l
            best_alpha = get_adaptive_alpha(args, model, model_long, all_train_sets, set_index)
            print('M*_e fusion weight delta={:.4f}'.format(best_alpha))
            model = linear_combination(args, model, model_long, best_alpha)

            model_short_old = copy.deepcopy(model_old)  # Theta_{t-1}^s for next step
            model_long = copy.deepcopy(model)            # Theta_t^l = fused model

        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[set_index]
        from reid.evaluators import extract_features
        features, _ = extract_features(model, test_loader, training_phase=set_index+1)
        print ("Save Features: ", len(features))
        torch.save({'features':features}, args.logs_dir+name+'_features.pth.tar')
        
        if args.trans_feat and set_index > 0:
            for each_data in training_set[0:set_index]:
                each_old_gallery = torch.load(args.logs_dir + str(each_data) + '_features.pth.tar')
                features_dict = each_old_gallery['features']
                feature_keys = list(features_dict.keys())
                feature_values = list(features_dict.values())
                feature_tensor = torch.stack([torch.tensor(val) for val in feature_values])
                feature_tensor = F.normalize(feature_tensor, p=2, dim=1)
                from collections import OrderedDict
                updated_features_dict = OrderedDict()
                model_trans.eval()                
                with torch.no_grad():
                    
                    for start_pos in range(0, feature_tensor.shape[0], args.batch_size):
                        end_pos = min(start_pos + args.batch_size, feature_tensor.shape[0])
                        batch_features = feature_tensor[start_pos:end_pos]
                        batch_features_trans = model_trans(batch_features)
                        batch_features_trans = F.normalize(batch_features_trans, p=2, dim=1)
                        batch_features_trans = best_alpha * batch_features_trans + (1 - best_alpha) * batch_features.cuda()
                        for i in range(start_pos, end_pos):
                            updated_features_dict[feature_keys[i]] = batch_features_trans[i - start_pos]
                    each_old_gallery['features'] = updated_features_dict
                    torch.save(each_old_gallery, args.logs_dir + str(each_data) + '_features.pth.tar')
                    print("Old Gallery Feature Updated to: ", args.logs_dir + str(each_data) + '_features.pth.tar')
            print('=================================================================================')
        if args.set_zero:
            model_trans = set_zero(model_trans)
            model_trans2 = set_zero(model_trans2)
        if set_index>0:
            test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, feats_dir=args.logs_dir)    
    print('finished')
def get_normal_affinity(x,Norm=100):
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix

def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    """
    LSTKC++ M*_e: compute delta via relation similarity (Eq.15).
    delta = 1 - mean_i( dot((R^s)_i, (R^l)_i) )
    Guaranteed in [0,1], robust to large domain gaps.
    model     = short-term new model Theta_t^s
    model_old = long-term old model  Theta_t^l
    """
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[set_index]
    features_all_new, _, _, _, _, _ = extract_features_voro(model, init_loader_new, get_mean_feature=True)
    features_all_old, _, _, _, _, _ = extract_features_voro(model_old, init_loader_new, get_mean_feature=True)

    features_all_new = torch.stack(features_all_new, dim=0)
    features_all_old = torch.stack(features_all_old, dim=0)

    Affin_new = get_normal_affinity(features_all_new)  # [n, n]
    Affin_old = get_normal_affinity(features_all_old)  # [n, n]

    # Eq.(15): delta = 1 - mean_i <R^s_i, R^l_i>
    dot_sim = (Affin_new * Affin_old).sum(dim=1).mean()
    alpha = float(1.0 - dot_sim)
    alpha = max(0.0, min(1.0, alpha))  # strict clamp to [0,1]
    return alpha



def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, model_trans, model_trans2, out_channel,
                                            writer, old_model_long=None, logger_res=None):
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])
    
    
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        add_num = sum([all_train_sets[i][1] for i in range(set_index)])

        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    

        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()

    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if set_index>0:
        for key, value in model_trans.named_params(model_trans):
            if not value.requires_grad:
                print('not requires_grad:', key)
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        for key, value in model_trans2.named_params(model_trans2):
            if not value.requires_grad:
                print('not requires_grad:', key)
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
  
    trainer = Trainer(cfg, args, model, model_trans, model_trans2, add_num + num_classes,  writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):

        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,
                      old_model_long=old_model_long,
                      )
        lr_scheduler.step()       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test:
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, feats_dir=args.logs_dir)                    
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model, model_trans, model_trans2

def test_model(model, all_train_sets, all_test_sets, set_index, logger_res=None, feats_dir=None):
    begin = 0
    evaluator = Evaluator(model)
        
    R1_all = []
    mAP_all = []
    names=''
    Results=''
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))

        train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                                 cmc_flag=True)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)
        names = names + name + '\t\t'
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()


    R1_all = []
    mAP_all = []
    names_unseen = ''
    Results_unseen = ''
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))

        R1, mAP = evaluator.evaluate_rfl(test_loader, dataset.query, dataset.gallery,
                                     cmc_flag=True, old_feat=feats_dir+name+'_features.pth.tar')
        R1_all.append(R1)
        mAP_all.append(mAP)
        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100)

    aver_mAP_unseen = torch.tensor(mAP_all).mean()
    aver_R1_unseen = torch.tensor(R1_all).mean()

    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    names_unseen = names_unseen + '|Average\t|'
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100)
    print(names_unseen)
    print(Results_unseen)
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        logger_res.append(names_unseen)
        logger_res.append(Results_unseen)
        logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP



def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new

def backward_rectification(args, model_short_old, model_long_old, init_loader):
    """
    LSTKC++ Algorithm 1: Old Knowledge Backward Rectification (M*_o).
    Searches optimal alpha in [0,1] using new data as unbiased reference.
    Returns updated long-term model Theta_t^l.
    """
    from reid.evaluators import extract_features
    from reid.evaluation_metrics import mean_ap
    from reid.evaluators import pairwise_distance

    best_mAP = 0.0
    best_alpha = 0.0

    for alpha_int in range(11):  # 0, 1, ..., 10  -> alpha = 0.0, 0.1, ..., 1.0
        alpha = alpha_int / 10.0
        model_fused = linear_combination(args, model_long_old, model_short_old, alpha)
        model_fused.eval()

        features, labels = extract_features(model_fused, init_loader)
        fnames = list(features.keys())
        feat_tensor = torch.stack([features[f] for f in fnames])
        label_list  = [int(labels[f]) for f in fnames]

        # self-ranking: use all features as both query and gallery
        feat_norm = F.normalize(feat_tensor, p=2, dim=1)
        dist = torch.pow(feat_norm, 2).sum(dim=1, keepdim=True).expand(len(fnames), len(fnames)) \
             + torch.pow(feat_norm, 2).sum(dim=1, keepdim=True).expand(len(fnames), len(fnames)).t()
        dist.addmm_(feat_norm, feat_norm.t(), beta=1, alpha=-2)
        dist_np = dist.cpu().numpy()

        label_arr = np.array(label_list)
        cam_arr   = np.zeros(len(label_list), dtype=np.int32)  # dummy cams
        mAP = mean_ap(dist_np, label_arr, label_arr, cam_arr, cam_arr)

        if mAP > best_mAP:
            best_mAP = mAP
            best_alpha = alpha

    print('M*_o backward rectification: best_alpha={:.1f}, best_mAP={:.4f}'.format(best_alpha, best_mAP))
    model_long_new = linear_combination(args, model_long_old, model_short_old, best_alpha)
    return model_long_new


def set_zero(model):
    '''old model '''
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        model_new_state_dict[k] = 0. * v
    
    model_new.load_state_dict(model_new_state_dict)
    return model_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
 
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])

    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')

    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--print-freq', type=int, default=200)
    
 
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xiong_project/data/PRID/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")    
    
    parser.add_argument('--trans_feat', action="store_false", help='if or not transform old features')
    parser.add_argument('--set_zero', action="store_true", help='if or not set 0 to transnet')
    parser.add_argument('--weight_trans', type=float, default=100, help='weight for transformation loss')
    parser.add_argument('--weight_anti', type=float, default=1, help='weight for anti_forget loss')
    parser.add_argument('--weight_discri', type=float, default=0.007, help='weight for anti_discrimination loss')
    parser.add_argument('--weight_transx', type=float, default=0.0005, help='weight for transformation_x loss')
    
    main()