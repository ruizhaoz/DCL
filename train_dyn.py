import sys, time, torch, random, argparse, json, math, shutil
import itertools
from collections import namedtuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from torch.distributions import Categorical

from starts import prepare_logger
from utils import get_model_infos
from model_dict import get_model_from_name
from get_dataset_with_transform import get_datasets
import wandb
import copy
from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time

print('done import')


def get_model_prefix(args):
    lmbda = '-'.join([str(args.KD_alpha), str(args.KD_alpha2), str(args.KD_alpha3), str(args.ema_decay), str(args.rand_seed)])
    prefix = '/home/rzhu/code/dcl/' + '-'.join(
        [args.dataset, args.method, args.model_name, args.loss_type, str(args.epochs), lmbda, '--'])

    return prefix


def m__get_prefix(args):
    lmbda = '-'.join([str(args.lmbda_adaptive), str(args.Ti), str(args.lmbda), str(args.lmbda_min), str(args.budget_g),
                      str(args.budget_g_ft), str(args.budget_g_stable), str(args.budget_g_ent), str(args.reg_ft),
                      str(args.lmbda_max_ent)])
    prefix = '-'.join(
        ['SD-', args.dataset, args.method, args.model_name, args.loss_type, str(args.epochs), lmbda, '--'])
    return prefix


def get_mlr(lr_scheduler):
    return lr_scheduler.optimizer.param_groups[0]['lr']


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parameter_copy(args, logger, teacher_model, model):
    decay = args.ema_decay

    with torch.no_grad():
        for ema_v, model_v in zip(teacher_model.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v)


def ema_update(args, logger, teacher_model, model):
    decay = args.ema_decay
    update_fn = lambda e, m: decay * e + (1. - decay) * m
    with torch.no_grad():
        for ema_v, model_v in zip(teacher_model.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(update_fn(ema_v, model_v))


def assign_update(args, logger, pre_model, current_model):
    # decay = args.ema_decay
    # update_fn = lambda e, m: decay * e + (1. - decay) * m

    with torch.no_grad():
        for pre_v, cur_v in zip(pre_model.state_dict().values(), current_model.state_dict().values()):
            pre_v.copy_(cur_v)


def get_lmbda_val(args, epoch):
    lmbda = args.lmbda
    l_min = args.lmbda_min
    l_max = args.lmbda  # _max
    l_max_ent = args.lmbda_max_ent  # _max

    if args.lmbda_adaptive == 1:
        args.lmbda_dual = l_max
        args.lmbda_dual_ent = l_max_ent
    elif args.lmbda_adaptive == 5:
        Ti = args.Ti  # args.epochs
        T = epoch % Ti  # args.epochs
        lmbda = l_min + (l_max - l_min) * (1 - math.cos(math.pi * T / Ti)) / 2

        lmbda_ent = l_min + (l_max_ent - l_min) * (1 - math.cos(math.pi * T / Ti)) / 2

        args.lmbda_dual = lmbda
        args.lmbda_dual_ent = lmbda_ent
    elif args.lmbda_adaptive == 6:
        lmbda = args.lmbda_dual

    elif args.lmbda_adaptive == 7:
        Ti = args.Ti  # args.epochs
        T = epoch % Ti  # args.epochs
        lmbda = l_min + (l_max - l_min) * (1 - math.cos(math.pi * T / Ti)) / 2
        lmbda_ent = l_min + (l_max_ent - l_min) * (1 - math.cos(math.pi * T / Ti)) / 2
        # args.budget_g_max
        # args.budget_g_min
        # args.constraint_val
        # args.lmbda_increase
        # Setup lmbda and lmbda_ent as per constraint value
        if args.constraint_val > args.budget_g_max:
            lmbda = max(0, lmbda + args.lmbda_increase)
            args.expq = 'max2min'
        elif args.constraint_val <= args.budget_g_max and args.constraint_val >= args.budget_g_min:
            if args.expq == 'max2min':
                lmbda = max(0, lmbda + args.lmbda_increase)
            elif args.expq == 'min2max':
                lmbda = max(0, lmbda - args.lmbda_increase)
            else:
                assert (1 == 2)
        else:  # args.constraint_val < args.budget_g_min
            lmbda = max(0, lmbda - args.lmbda_increase)
            args.expq = 'min2max'
        # lmbda = max(0, lmbda + )

        args.lmbda_dual = lmbda
        args.lmbda_dual_ent = lmbda_ent

    elif args.lmbda_adaptive == 8:
        Ti = args.Ti  # args.epochs
        T = epoch % Ti  # args.epochs
        lmbda = l_max + (l_min - l_max) * (1 - math.cos(math.pi * T / Ti)) / 2

        lmbda_ent = l_max_ent + (l_min - l_max_ent) * (1 - math.cos(math.pi * T / Ti)) / 2

        args.lmbda_dual = lmbda
        args.lmbda_dual_ent = lmbda_ent

    else:
        assert (1 == 2)

    # print('lmbda = ', lmbda)
    return lmbda


def take_a_step_optim_model(optimizer, model, new_loader, criterion):
    inputs, targets = next(iter(new_loader))
    optimizer.zero_grad()

    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    _, logits, _ = model(inputs)
    loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def get_KD_loss(p, q, temperature):
    log_student = F.log_softmax(p / temperature, dim=1)
    sof_teacher = F.softmax(q / temperature, dim=1)
    KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean")

    return KD_loss

def get_KD_loss_tri(p, q, temperature):
    log_student = F.log_softmax(p / temperature, dim=1)
    mq = (q+p)/2
    sof_teacher = F.softmax(mq / temperature, dim=1)
    KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean")

    return KD_loss

def get_trangular_loss_sq(p, q):
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight =torch.sqrt (torch.square(p.detach()) + torch.square(q.detach()))
    MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_trangular_loss_sqv1(p, q):
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight =torch.sqrt(torch.square(p) + torch.square(q))
    MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_trangular_loss_sqv2(p, q):
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight =torch.sqrt(torch.square(p.detach()) + torch.square(q.detach()))
    MSE_weight = F.normalize(MSE_weight, p=1, dim=1)
    MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_softmse_loss(p, q, temperature):
    p = F.softmax(p / temperature, dim=1)
    q = F.softmax(q / temperature, dim=1)
    MSE_loss = F.mse_loss(p, q, reduction='mean')
    # MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    # MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_softmse_loss_v2(p, q, temperature):
    p = F.softmax(p / temperature, dim=1)
    q = F.softmax(q / temperature, dim=1)
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    MSE_weight = F.normalize(MSE_weight, p=1, dim=1)
    MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_softmse_loss_v3(p, q, temperature):
    p = F.softmax(p / temperature, dim=1)
    q = F.softmax(q / temperature, dim=1)
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    MSE_weight = F.normalize(MSE_weight, p=1, dim=1)
    MSE_loss = torch.mean(MSE_loss * MSE_weight )
    return MSE_loss

def get_lrelumse_loss(p, q, negative_slope=0.01):
    p = F.leaky_relu(p, negative_slope= negative_slope)
    q = F.leaky_relu(q, negative_slope= negative_slope)
    MSE_loss = F.mse_loss(p, q, reduction='mean')
    # MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    # MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_trangular_loss(p, q):
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_trangular_loss_v2(p, q):
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    MSE_weight = F.normalize(MSE_weight, p=1, dim=1)
    MSE_loss = torch.mean(MSE_loss / MSE_weight )
    return MSE_loss

def get_trangular_loss_v3(p, q):
    MSE_loss = F.mse_loss(p, q, reduction='none')
    MSE_weight = torch.abs(p.detach()) + torch.abs(q.detach())
    MSE_weight = F.normalize(MSE_weight, p=1, dim=1)
    MSE_loss = torch.mean(MSE_loss * MSE_weight )
    return MSE_loss


def info_nce_loss(features, temperature=1):
    temperature = args.c_temperature
    bsize = features.shape[0]//2
    labels = torch.cat([torch.arange(bsize) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device =features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def get_entropy(args, p):
    # '''
    C = np.log(args.class_num)
    probs = F.softmax(p, dim=-1)
    entropy = - torch.sum(probs * torch.log(probs + 1e-5) / C) / len(p)
    return entropy


def get_entropy_loss(args, p, q, r):
    entropy_p = get_entropy(args, p)
    entropy_q = get_entropy(args, q)
    entropy_r = get_entropy(args, r)
    entropy = 0.33 * (entropy_p + entropy_q + entropy_r)
    # print('entropy = ', entropy)
    return entropy

# def classwise_acc(logits, target):
#

def update_lmbda_dual_val(args, logger, inputs, aug1, aug2, targets, network, temperature):
    with torch.no_grad():
        _, logits, _ = network(inputs)
        p_ft, p_logits, _ = network(aug1)
        q_ft, q_logits, _ = network(aug2)

        Ent_loss = None
        if args.loss_type == 'KL':
            KD_loss = get_KD_loss(p_logits, q_logits, temperature)  # + get_KD_loss(q_logits, p_logits, temperature)
        elif args.loss_type == 'KLS':
            KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)
        elif args.loss_type == 'MSE':
            KD_loss = F.mse_loss(p_logits, q_logits)
        else:
            print('Loss type undefined.. ', args.loss_type)
        if Ent_loss is None: Ent_loss = 0. * KD_loss
    C = np.log(args.class_num)
    # args.lmbda_dual = ( F.relu(  args.lmbda_dual  + ( KD_loss - args.budget_g )**2 ) ).item()
    # args.lmbda_dual_ent = ( F.relu(  args.lmbda_dual_ent  + ( (Ent_loss - args.budget_g_ent)*C )**2 ) ).item()

    args.lmbda_dual = (F.relu(args.lmbda_dual + (KD_loss - args.budget_g))).item()
    args.lmbda_dual_ent = (F.relu(args.lmbda_dual_ent + ((Ent_loss - args.budget_g_ent) * C))).item()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cifar_100_train_eval_loop(pre_model_list, new_loader, args, logger, epoch, optimizer, scheduler, teacher_optimizer, teacher_scheduler,
                              teacher, network,local_par, xloader, criterion, batch_size, base=False, margin=False, mode='eval',error=0):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    losses = AverageMeter('Loss', ':.4e')

    min_entropy_constraint = AverageMeter('MinEnt', ':.4e')
    algorithmic_stability_constraint = AverageMeter('KLxxp', ':.4e')
    stable_constraint = AverageMeter('KLwwp', ':.4e')
    ft_constraint = AverageMeter('Ft', ':.4e')

    constraint = AverageMeter('Cnstrnt', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    alpha = args.KD_alpha
    alpha2 = args.KD_alpha2
    alpha3 = args.KD_alpha3
    temperature = args.KD_temperature

    N = len(xloader) * batch_size

    teacher.eval()
    if mode == 'eval':
        network.eval()
        teacher.eval()
    else:
        network.train()
        teacher.train()

    burn_in_epoch = -1  # 2 #10 #2 #10

    if epoch == burn_in_epoch:
        args.lmbda_dual = 0
        args.lmbda_dual_ent = 0

    # for epoch in range(epochs):
    progress = ProgressMeter(
        logger,
        len(xloader),
        [losses, top1, top5, constraint, min_entropy_constraint, algorithmic_stability_constraint, stable_constraint,
         ft_constraint],
        prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets, aug1, aug2) in enumerate(xloader):
        if mode == 'train':

            optimizer.zero_grad()
            teacher_optimizer.zero_grad()

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)


        if mode == 'train':
            r = np.random.rand(1)
            if args.method == 'CutMix_MSE-v1' and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                # compute output
                features, logits, _ = network(inputs)
                loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
            else:
                features, logits, _ = network(inputs)
                loss = criterion(logits, targets)
        else:
            with torch.no_grad():
                features, logits, _ = network(inputs)
                loss = criterion(logits, targets)
        KD_loss = Ent_loss = 0 * loss
        _constraint = loss

        if mode == 'train':
            if args.method in ['DCL']:
                '''
                tloss is the loss for companion model. It can use any type of distance loss.
                '''
                _, t_logits, _ = teacher(inputs)
                KD_loss = F.mse_loss(logits, t_logits.detach())
                loss = loss + alpha  * KD_loss
                tloss = F.mse_loss( (1 - args.ema_decay) * logits.detach() + args.ema_decay * t_logits.detach() , t_logits)  # method 1 seperate targets
                tloss.backward()
                teacher_optimizer.step()

            elif args.method in ['DCL-decoupled']:
                _, t_logits, _ = teacher(inputs)
                tloss = F.mse_loss( (1 - args.ema_decay) * logits.detach() + args.ema_decay * t_logits.detach() , t_logits)  # method 1 seperate targets

                tloss.backward()
                teacher_optimizer.step()
        else:
            KD_loss = Ent_loss = 0 * loss
            Ft_loss = Stable_loss = 0 * loss


        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        #### how the models is updated
        if mode == 'train':
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        constraint.update(_constraint.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == 'train':
            scheduler.step(epoch)
            teacher_scheduler.step(epoch)



            if args.method == 'CE':
                ema_update(args, logger, teacher, network)

        if (i % args.print_freq == 0) or (i == len(xloader) - 1):
            progress.display(i)
    return losses.avg, top1.avg, top5.avg, min_entropy_constraint.avg, algorithmic_stability_constraint.avg, stable_constraint.avg, ft_constraint.avg


def get_updated_val(prev_budget, constraint):
    if prev_budget > constraint:
        budget_g = constraint / 2.
    else:
        budget_g = constraint
    return budget_g


def main(args):
    args.save_dir = args.save_dir + m__get_prefix(args)
    print(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
    print(xshape)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    new_train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    args.class_num = class_num
    logger = prepare_logger(args)

    model = get_model_from_name(class_num, args.model_name, args.dataset)
    model = model.cuda()

    teacher_model = get_model_from_name(class_num, args.model_name, args.dataset)
    teacher_model = copy.deepcopy(model)
    teacher_model = teacher_model.cuda()
    #
    # assign_update(args, logger, teacher_model, model)

    # pre_model = get_model_from_name(class_num, args.model_name, args.dataset)
    # pre_model = pre_model.cuda()
    # assign_update(args, logger, pre_model, model)

    pre_model_list = []
    for i in range(args.interval):
        pre_m= get_model_from_name(class_num, args.model_name, args.dataset)
        pre_m = pre_m.cuda()
        assign_update(args, logger, pre_m, model)
        pre_model_list.append(pre_m)

    # model_func = lambda : client_model(model_name)
    n_par = len(get_mdl_params([model])[0])
    # local_par = np.zeros((1,n_par)).astype('float32')
    local_par = np.zeros((n_par)).astype('float32')
    local_par = torch.tensor(local_par, dtype=torch.float32).cuda()

    # assign_update(args, logger, teacher_model, model)

    # if args._ckpt != "" and len(args._ckpt) > 3:
    #     state = torch.load(args._ckpt)
    #     model.load_state_dict(state['base_state_dict'])
    #     teacher_model.load_state_dict(state['base_state_dict'])

    teacher_optimizer = torch.optim.SGD(teacher_model.parameters(), args.lr_t, momentum=args.momentum, weight_decay=args.wd)
    # teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), args.lr_t,weight_decay=args.wd)
    teacher_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_optimizer, args.epochs)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.6, last_epoch=-1)



    flop, param = get_model_infos(model, xshape)
    logger.log("model information : {:}".format(model.get_message()))
    logger.log("-" * 50)
    logger.log(
        "[Base]Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
            param, flop, flop / 1e3
        )
    )
    logger.log("-" * 50)
    logger.log("train_data : {:}".format(train_data))
    logger.log("valid_data : {:}".format(valid_data))

    epoch = -1
    val_loss, val_acc1, val_acc5, min_entropy, algorithmic_stability, stable_const, ft_const = cifar_100_train_eval_loop(pre_model_list,
        new_train_loader, args, logger, epoch, optimizer, scheduler, teacher_optimizer, teacher_scheduler, teacher_model, model,local_par,
        # train_loader, criterion, args.eval_batch_size, base=True, mode='eval' )
        valid_loader, criterion, args.eval_batch_size, base=True, mode='eval')
    best_acc = val_acc1
    best_teacher_acc = val_acc1



    best_state_dict = model.state_dict()
    stage = 0
    error = 0
    for epoch in range(args.epochs):
        trn_loss, trn_acc1, trn_acc5, trn_min_entropy, trn_algorithmic_stability, trn_stable, trn_ft = cifar_100_train_eval_loop(pre_model_list,
            new_train_loader, args, logger, epoch, optimizer, scheduler, teacher_optimizer, teacher_scheduler, teacher_model,
            model,local_par, train_loader, criterion, args.batch_size, base=True, mode='train', error=error)
        val_loss, val_acc1, val_acc5, val_min_entropy, val_algorithmic_stability, val_stable, val_ft = cifar_100_train_eval_loop(pre_model_list,
            new_train_loader, args, logger, epoch, optimizer, scheduler, teacher_optimizer, teacher_scheduler, teacher_model,
            model,local_par, valid_loader, criterion, args.eval_batch_size, base=True, mode='eval')

        is_best = False
        if val_acc1 > best_acc:
            best_acc = val_acc1
            is_best = True
            best_state_dict = model.state_dict()

        _, teacher_acc1, _, teacher_val_min_entropy, teacher_val_algorithmic_stability, teacher_stable, teacher_ft = cifar_100_train_eval_loop(pre_model_list,
            new_train_loader, args, logger, epoch, optimizer, scheduler, teacher_optimizer, teacher_scheduler, teacher_model,
            teacher_model,local_par, valid_loader, criterion, args.eval_batch_size, base=True, mode='eval')
        if teacher_acc1 > best_teacher_acc:
            best_teacher_acc = teacher_acc1
        wandb.log({"val acc": val_acc1, "val teacher acc": teacher_acc1})
        wandb.log({"best val acc": best_acc, "best val teacher acc": best_teacher_acc})

        state = {
            'epoch': epoch + 1,
            'base_state_dict': model.state_dict(),
            'teacher_state_dict': teacher_model.state_dict(),
            'best_acc': best_acc,
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if args.method in ['SD', 'our-SD']:
            state['teacher'] = teacher_model.state_dict()

        save_checkpoint(state, is_best, prefix=get_model_prefix(args) + 'stage-' + str(stage) + '-')

        # if epoch in need_save:
        #     torch.save(state, '/home/ruizhu/code/DCL/save/'+'checkpoint-'+str(epoch)+'.pth.tar')

        lmbda = get_lmbda_val(args, epoch)
        logger.log('Stage=' + str(stage) + '\t\t LR=' + str(get_mlr(scheduler)) + ' -- best acc so far ' + str(
            best_acc) + ' -- teacher acc ' + str(best_teacher_acc) + ' -- lmbda = ' + str(
            lmbda) + ' -- l-adap=' + str(args.lmbda_adaptive))

    model.load_state_dict(best_state_dict)
    teacher_model.load_state_dict(best_state_dict)

    val_loss, val_acc1, val_acc5, min_entropy, algorithmic_stability, stable_const, ft_const = cifar_100_train_eval_loop(pre_model_list,
        new_train_loader, args, logger, epoch, optimizer, scheduler, teacher_optimizer, teacher_scheduler, teacher_model, model,local_par,
        # train_loader, criterion, args.eval_batch_size, base=True, mode='eval' )
        valid_loader, criterion, args.eval_batch_size, base=True, mode='eval')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Distillation CIFAR-10/100 model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     )
    parser.add_argument(
        "--model_name", type=str,
        default='ResNet18',
        help="The path to the model configuration"
    )
    parser.add_argument('--budget_g_max', type=float, default=0.1, help='learning rate for a single GPU')
    parser.add_argument('--budget_g_min', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_increase', type=float, default=0.1, help='learning rate for a single GPU')
    parser.add_argument('--constraint_val', type=float, default=0.1, help='learning rate for a single GPU')

    parser.add_argument('--budget_g', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--budget_g_ft', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--budget_g_stable', type=float, default=0.0, help='learning rate for a single GPU')
    parser.add_argument('--budget_g_ent', type=float, default=0.0, help='learning rate for a single GPU')

    parser.add_argument('--lmbda_dual_ent', type=float, default=0.4, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_max_ent', type=float, default=0.4, help='learning rate for a single GPU')
    parser.add_argument('--reg_ft', type=float, default=0.5, help='learning rate for a single GPU')

    parser.add_argument('--lmbda', type=float, default=0.4, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_dual', type=float, default=0.4, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_min', type=float, default=0.01, help='learning rate for a single GPU')
    parser.add_argument('--lmbda_adaptive', type=int, default=5, help='learning rate for a single GPU')
    parser.add_argument('--Ti', type=int, default=100, help='number of epochs to train')

    parser.add_argument('--ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')
    parser.add_argument('--rho', type=float, default=1,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # parser.add_argument(
    #     "--_ckpt", type=str,
    #     default='',
    #     help="The path to the model checkpoint"
    # )

    parser.add_argument(
        "--KD_alpha", type=float, default=0.9,
        help="The alpha parameter in knowledge distillation."
    )

    parser.add_argument(
        "--KD_alpha2", type=float, default=0.9,
        help="The alpha parameter in knowledge distillation."
    )
    parser.add_argument(
        "--KD_alpha3", type=float, default=0.9,
        help="The alpha parameter in knowledge distillation."
    )
    parser.add_argument(
        "--KD_temperature",
        type=float,
        default=1,
        help="The temperature parameter in knowledge distillation.",
    )

    parser.add_argument(
        "--c_temperature",
        type=float,
        default=4,
        help="The temperature parameter in knowledge distillation.",
    )

    parser.add_argument(
        "--KD_temperature_s",
        type=float,
        default=3,
        help="The temperature parameter in knowledge distillation.",
    )



    parser.add_argument("--method", type=str, default='CE', help="The method name. (CE, SD, our-SD).")
    parser.add_argument("--loss_type", type=str, default='CE', help="The method name. (CE, SD, our-SD).")

    # Data Generation
    parser.add_argument("--dataset", type=str, default='aug-cifar100', help="The dataset name.")
    # parser.add_argument("--data_path", type=str, default='/Data/rzhu/ImageNet-1000/', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='/Data/rzhu/cifar-100-python', help="The dataset name.")
    parser.add_argument(
        "--cutout_length", type=int, default=16, help="The cutout length, negative means not use."
    )

    # Printing
    parser.add_argument(
        "--print_freq", type=int, default=100, help="print frequency (default: 200)"
    )
    parser.add_argument(
        "--print_freq_eval",
        type=int,
        default=100,
        help="print frequency (default: 200)",
    )
    # Checkpoints
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1,
        help="evaluation frequency (default: 200)",
    )
    parser.add_argument(
        "--save_dir", type=str, help="Folder to save checkpoints and log.",
        default='./logs/',
    )
    # Acceleration
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of data loading workers (default: 8)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=3407, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    # add_shared_args(parser)

    # Optimization options
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=200, help="Batch size for training."
    )
    parser.add_argument('--log-dir', default='./log', help='tensorboard log directory')
    parser.add_argument('--checkpoint-dir', default='./checkpoint',
                        help='checkpoint file format')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--lr_t', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--lrelu', type=float, default=0.01,
                        help='learning rate for a single GPU')
    parser.add_argument('--interval', type=int, default=1,
                            help='learning rate for a single GPU')
    parser.add_argument('--inner_iter', type=int, default=5,
                            help='learning rate for a single GPU')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')

    parser.add_argument('--beta', default=1, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=1, type=float,
                        help='cutmix probability')

    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"
    run=wandb.init(project="cifar_DCL_resnet18",
                     config={
            "method":args.method,
            "model_name":args.model_name,
            "alpha": args.KD_alpha,
            "ema_decay": args.ema_decay,
            "alpha2": args.KD_alpha2,
            "alpha3": args.KD_alpha3,
            "temp": args.KD_temperature,
            "rand_seed":args.rand_seed,
            # "warm_up":args.warmup
        })
    main(args)


