
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

from log_utils import AverageMeter, ProgressMeter, time_string, convert_secs2time

def get_model_prefix( args ):
    lmbda = '-'.join([ str(args.lmbda_adaptive), str(args.Ti), str(args.lmbda), str(args.lmbda_min), str(args.budget_g), str(args.budget_g_ft), str(args.budget_g_stable), str(args.budget_g_ent), str( args.reg_ft ), str(args.lmbda_max_ent) ]) 
    prefix = './models/' + '-'.join( [ args.dataset, args.method, args.model_name, args.loss_type, str(args.epochs), lmbda, '--' ] )
    return prefix

def m__get_prefix( args ):
    lmbda = '-'.join([ str(args.lmbda_adaptive), str(args.Ti), str(args.lmbda), str(args.lmbda_min), str(args.budget_g), str(args.budget_g_ft), str(args.budget_g_stable), str(args.budget_g_ent), str( args.reg_ft ), str(args.lmbda_max_ent)  ]) 
    prefix = '-'.join( [ 'SD-', args.dataset, args.method, args.model_name, args.loss_type, str(args.epochs), lmbda, '--' ] )
    return prefix

def get_mlr(lr_scheduler):
     return lr_scheduler.optimizer.param_groups[0]['lr']

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+'model_best.pth.tar')

def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parameter_copy( args, logger, avg_model, model ):
    decay = args.ema_decay

    with torch.no_grad():
        for ema_v, model_v in zip( avg_model.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v)


def ema_update( args, logger, avg_model, model ):
    decay = args.ema_decay
    update_fn = lambda e, m: decay * e + (1. - decay) * m

    with torch.no_grad():
        for ema_v, model_v in zip( avg_model.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(update_fn(ema_v, model_v))


def get_lmbda_val( args, epoch ):
    lmbda = args.lmbda  
    l_min = args.lmbda_min
    l_max = args.lmbda #_max
    l_max_ent = args.lmbda_max_ent #_max

    if args.lmbda_adaptive == 1:
        args.lmbda_dual = l_max
        args.lmbda_dual_ent = l_max_ent 
    elif args.lmbda_adaptive == 5:
        Ti = args.Ti #args.epochs
        T = epoch % Ti #args.epochs
        lmbda = l_min + (l_max - l_min) * (1 - math.cos(math.pi * T / Ti) ) / 2

        lmbda_ent = l_min + (l_max_ent - l_min) * (1 - math.cos(math.pi * T / Ti) ) / 2

        args.lmbda_dual = lmbda
        args.lmbda_dual_ent = lmbda_ent
    elif args.lmbda_adaptive == 6:
        lmbda = args.lmbda_dual

    elif args.lmbda_adaptive == 7:

        Ti = args.Ti #args.epochs
        T = epoch % Ti #args.epochs
        lmbda = l_min + (l_max - l_min) * (1 - math.cos(math.pi * T / Ti) ) / 2

        lmbda_ent = l_min + (l_max_ent - l_min) * (1 - math.cos(math.pi * T / Ti) ) / 2

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
                assert(1==2)
        else: #         args.constraint_val < args.budget_g_min
            lmbda = max(0, lmbda - args.lmbda_increase)
            args.expq = 'min2max'
        #lmbda = max(0, lmbda + )

        args.lmbda_dual = lmbda
        args.lmbda_dual_ent = lmbda_ent

    elif args.lmbda_adaptive == 8:
        Ti = args.Ti #args.epochs
        T = epoch % Ti #args.epochs
        lmbda = l_max + (l_min - l_max) * (1 - math.cos(math.pi * T / Ti) ) / 2

        lmbda_ent = l_max_ent + (l_min - l_max_ent) * (1 - math.cos(math.pi * T / Ti) ) / 2

        args.lmbda_dual = lmbda
        args.lmbda_dual_ent = lmbda_ent

    else:
        assert(1==2)

    #print('lmbda = ', lmbda)
    return lmbda


def take_a_step_optim_model( optimizer, model, new_loader, criterion ):
    inputs, targets = next( iter(new_loader) )
    optimizer.zero_grad()

    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    _, logits, _ = model(inputs)
    loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()

def get_KD_loss(p, q, temperature):
    log_student = F.log_softmax(p / temperature, dim=1)
    sof_teacher = F.softmax(q / temperature, dim=1)
    KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean") 

    return KD_loss


def get_entropy(args, p):
    #'''
    C = np.log(args.class_num)
    probs = F.softmax(p, dim=-1)
    entropy = - torch.sum( probs * torch.log(probs + 1e-5)/C ) / len(p)
    #entropy = entropy #* C
    #'''
    #entropy = Categorical( probs=F.softmax(p, dim=-1) ).entropy() #.unsqueeze(1)
    #entropy = torch.mean( entropy )
    return entropy

def get_entropy_loss(args, p, q, r):
    entropy_p = get_entropy(args, p)
    entropy_q = get_entropy(args, q)
    entropy_r = get_entropy(args, r)
    entropy = 0.33 * (entropy_p + entropy_q + entropy_r)
    #print('entropy = ', entropy)
    return entropy


def update_lmbda_dual_val( args, logger, inputs, aug1, aug2, targets, network, temperature ):
    with torch.no_grad():
            _, logits, _ = network(inputs)
            p_ft, p_logits, _ = network(aug1)
            q_ft, q_logits, _ = network(aug2)

            Ent_loss = None
            if args.loss_type == 'KL':
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) #+ get_KD_loss(q_logits, p_logits, temperature)
            elif args.loss_type == 'KLS':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)
            elif args.loss_type == 'MSE':   
                KD_loss = F.mse_loss( p_logits, q_logits ) 
            elif args.loss_type == 'MSES':   
                KD_loss = F.mse_loss( logits, q_logits ) + F.mse_loss( logits, p_logits )
            elif args.loss_type == 'KLS-MinEnt':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)
                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature) + .5 * F.mse_loss( p_ft, q_ft )
                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            else:
                print('Loss type undefined.. ', args.loss_type)
            if Ent_loss is None: Ent_loss = 0. * KD_loss
    C = np.log(args.class_num)
    #args.lmbda_dual = ( F.relu(  args.lmbda_dual  + ( KD_loss - args.budget_g )**2 ) ).item()
    #args.lmbda_dual_ent = ( F.relu(  args.lmbda_dual_ent  + ( (Ent_loss - args.budget_g_ent)*C )**2 ) ).item()

    args.lmbda_dual = ( F.relu(  args.lmbda_dual  + ( KD_loss - args.budget_g ) ) ).item()
    args.lmbda_dual_ent = ( F.relu(  args.lmbda_dual_ent  + ( (Ent_loss - args.budget_g_ent)*C ) ) ).item()


def get_constraint_vals(aug1, aug2, args, network, teacher, logits, p_logits, q_logits, temperature, p_ft, q_ft, conv_p_ft, conv_q_ft):
            KD_loss = torch.zeros((1,), device=logits.device)
            Ent_loss = None
            Ft_loss = None
            Stable_loss = None

            if args.loss_type == 'KL':
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) #+ get_KD_loss(q_logits, p_logits, temperature)
            elif args.loss_type == 'KLS':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)
            elif args.loss_type == 'MSE':   
                KD_loss = F.mse_loss( p_logits, q_logits ) 
            elif args.loss_type == 'MSES':   
                KD_loss = F.mse_loss( logits, q_logits ) + F.mse_loss( logits, p_logits )
            elif args.loss_type == 'KLS-MinEnt':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)
                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature) + args.reg_ft * F.mse_loss( p_ft, q_ft )
                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft2':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature) +  args.reg_ft  * F.mse_loss( p_ft, q_ft )
                #Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft3':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature) +  args.reg_ft  * ( F.mse_loss( p_ft, q_ft ) + F.mse_loss(conv_p_ft, conv_q_ft) )
                #Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft4':   
                KD_loss = args.reg_ft  * ( F.mse_loss( p_ft, q_ft ) )
                #Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft5':   

                KD_loss =  args.reg_ft  * F.mse_loss( p_ft, q_ft )

                with torch.no_grad():
                    tp_ft, tp_logits, tp_all_ft = teacher(aug1)
                    tq_ft, tq_logits, tq_all_ft = teacher(aug2)

                #KD_loss = KD_loss + 0.5 * args.reg_ft  * ( F.mse_loss( p_ft, tq_ft ) +  F.mse_loss( q_ft, tp_ft ) )
                KD_loss = KD_loss + 0.5 * args.reg_ft * ( F.mse_loss( p_ft, tp_ft ) +  F.mse_loss( q_ft, tq_ft ) )

                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft6':   
                KD_loss = args.reg_ft  * ( F.mse_loss( p_ft, q_ft ) )
                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft7':   

                #KD_loss =  args.reg_ft  * F.mse_loss( p_ft, q_ft )
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)

                with torch.no_grad():
                    tp_ft, tp_logits, tp_all_ft = teacher(aug1)
                    tq_ft, tq_logits, tq_all_ft = teacher(aug2)

                KD_loss = KD_loss + 0.5 * args.reg_ft * ( get_KD_loss(p_logits, tp_logits, temperature) + get_KD_loss(q_logits, tq_logits, temperature)  )
                #KD_loss = KD_loss + 0.5 * args.reg_ft  * ( F.mse_loss( p_ft, tq_ft ) +  F.mse_loss( q_ft, tp_ft ) )
                #KD_loss = KD_loss + 0.5 * args.reg_ft * ( F.mse_loss( p_ft, tp_ft ) +  F.mse_loss( q_ft, tq_ft ) )

                Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
 
            elif args.loss_type == 'KLS-MinEnt-Ft8':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)

                with torch.no_grad():
                    tp_ft, tp_logits, tp_all_ft = teacher(aug1)
                    tq_ft, tq_logits, tq_all_ft = teacher(aug2)

                Ft_loss = F.mse_loss( p_ft, q_ft ) + F.mse_loss( p_ft, tp_ft ) + F.mse_loss( q_ft, tq_ft ) 
                Stable_loss = get_KD_loss(p_logits, tp_logits, temperature) + get_KD_loss(q_logits, tq_logits, temperature) 
                #KD_loss = KD_loss + 0.5 * args.reg_ft  * ( F.mse_loss( p_ft, tq_ft ) +  F.mse_loss( q_ft, tp_ft ) )
                #KD_loss = KD_loss + 0.5 * args.reg_ft * ( F.mse_loss( p_ft, tp_ft ) +  F.mse_loss( q_ft, tq_ft ) )

                #Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)
            elif args.loss_type == 'KLS-MinEnt-Ft9':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)

                with torch.no_grad():
                    tp_ft, tp_logits, tp_all_ft = teacher(aug1)
                    tq_ft, tq_logits, tq_all_ft = teacher(aug2)

                Stable_loss = get_KD_loss(p_logits, tp_logits, temperature) + get_KD_loss(q_logits, tq_logits, temperature) 
            elif args.loss_type == 'KLS-MinEnt-Ft10':   
                KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature)
            elif args.loss_type == 'KLS-MinEnt-Ft11':   
                with torch.no_grad():
                    tp_ft, tp_logits, tp_all_ft = teacher(aug1)
                    tq_ft, tq_logits, tq_all_ft = teacher(aug2)

                Stable_loss = get_KD_loss(p_logits, tp_logits, temperature) + get_KD_loss(q_logits, tq_logits, temperature) 
                KD_loss = 0. * Stable_loss
            else:

                print('Loss type undefined.. ', args.loss_type)
            #KD_loss = F.cross_entropy( p_logits, targets ) + F.cross_entropy( q_logits, targets )
            if Ent_loss is None: Ent_loss = 0. * KD_loss
            if Ft_loss is None: Ft_loss = 0. * KD_loss
            if Stable_loss is None: Stable_loss = 0. * KD_loss

            return KD_loss, Ent_loss, Ft_loss, Stable_loss


def cifar_100_train_eval_loop( new_loader, args, logger, epoch, optimizer, scheduler, avg_optimizer, avg_scheduler, teacher, network, xloader, criterion, batch_size, base=False, margin=False, mode='eval' ):

    losses = AverageMeter('Loss', ':.4e')

    min_entropy_constraint = AverageMeter('MinEnt', ':.4e')
    algorithmic_stability_constraint = AverageMeter('KLxxp', ':.4e')
    stable_constraint = AverageMeter('KLwwp', ':.4e')
    ft_constraint = AverageMeter('Ft', ':.4e')

    constraint = AverageMeter('Cnstrnt', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    alpha = args.KD_alpha
    temperature = args.KD_temperature

    N = len(xloader) * batch_size

    teacher.eval()
    if mode == 'eval': 
        network.eval()
    else:
        network.train()

    burn_in_epoch = -1 #2 #10 #2 #10

    if epoch == burn_in_epoch:
        args.lmbda_dual = 0
        args.lmbda_dual_ent = 0

    #for epoch in range(epochs):    
    progress = ProgressMeter(
            logger,
            len(xloader),
            [losses, top1, top5, constraint, min_entropy_constraint, algorithmic_stability_constraint, stable_constraint, ft_constraint],
            prefix="[{}] E: [{}]".format(mode.upper(), epoch))

    for i, (inputs, targets, aug1, aug2) in enumerate(xloader):
        if mode == 'train':
            if args.method == 'our-SD':
                '''if epoch > burn_in_epoch:
                    teacher.train()
                    take_a_step_optim_model( avg_optimizer, teacher, new_loader, criterion )
                    take_a_step_optim_model( optimizer, network, new_loader, criterion )
                    teacher.eval()'''
                pass    
            else:        
                inputs = aug1 

            optimizer.zero_grad()

        inputs = inputs.cuda(non_blocking=True)
        aug1 = aug1.cuda(non_blocking=True)
        aug2 = aug2.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mode == 'train':
            features, logits, _ = network(inputs)
        else:    
            with torch.no_grad():
                features, logits, _ = network(inputs)
        loss = criterion(logits, targets)
        KD_loss = Ent_loss = 0 * loss
        _constraint = loss 

        if mode == 'train':
          if args.method in ['SD']:
            with torch.no_grad():
                _, t_logits, _ = teacher(inputs)

            log_student = F.log_softmax(logits / temperature, dim=1)
            sof_teacher = F.softmax(t_logits / temperature, dim=1)
            KD_loss = F.kl_div(log_student, sof_teacher, reduction="batchmean") 

            loss = (1.-alpha) * loss + alpha * temperature * temperature * KD_loss

          elif args.method in ['CE']:
            KD_loss = Ent_loss = 0 * loss
            Ft_loss = Stable_loss = 0 * loss
          elif args.method in ['our-SD'] and (epoch > burn_in_epoch):
            #with torch.no_grad():
            #    _, t_logits, _ = teacher(inputs)

            p_ft, p_logits, p_all_ft = network(aug1)
            q_ft, q_logits, q_all_ft = network(aug2)
            conv_p_ft, conv_q_ft = p_all_ft[-1], q_all_ft[-1]

            KD_loss, Ent_loss, Ft_loss, Stable_loss = get_constraint_vals(aug1, aug2, args, network, teacher, logits, p_logits, q_logits, temperature, p_ft, q_ft, conv_p_ft, conv_q_ft)

            args.constraint_val = KD_loss.item()

            C = np.log(args.class_num)
            lmbda = get_lmbda_val( args, epoch )
            _constraint = lmbda * alpha * temperature * temperature * ( KD_loss - args.budget_g )**2 \
                    + args.lmbda_dual_ent * alpha * ( ( Ent_loss - args.budget_g_ent ) * C )**2 \
                    + lmbda * alpha * ( Ft_loss - args.budget_g_ft )**2 \
                    + lmbda * alpha * temperature * temperature * (Stable_loss - args.budget_g_stable)**2 
            #print('Ent_loss ', Ent_loss.item() )

            #_constraint = lmbda * alpha * temperature * temperature * ( KD_loss - args.budget_g ) \
            #        + args.lmbda_dual_ent * alpha * ( ( Ent_loss - args.budget_g_ent ) * C )

            #_constraint = lmbda * alpha * temperature * temperature * F.relu( KD_loss - args.budget_g )
            #_constraint = lmbda * alpha * temperature * temperature * ( KD_loss - args.budget_g )
            loss = (1.-alpha) * loss + _constraint 
        else:
          if args.method in ['CE']:
            KD_loss = Ent_loss = 0 * loss
            Ft_loss = Stable_loss = 0 * loss
          else:
            with torch.no_grad():
              p_ft, p_logits, p_all_ft = network(aug1)
              q_ft, q_logits, q_all_ft = network(aug2)
            conv_p_ft, conv_q_ft = p_all_ft[-1], q_all_ft[-1]

            KD_loss, Ent_loss, Ft_loss, Stable_loss = get_constraint_vals(aug1, aug2, args, network, teacher, logits, p_logits, q_logits, temperature, p_ft, q_ft, conv_p_ft, conv_q_ft)
            #KD_loss = get_KD_loss(p_logits, q_logits, temperature) + get_KD_loss(q_logits, p_logits, temperature) + args.reg_ft  *   ( F.mse_loss( p_ft, q_ft ) + F.mse_loss(conv_p_ft, conv_q_ft) ) 
            #Ent_loss = get_entropy_loss(args, p_logits, q_logits, logits)

        prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))

        if mode == 'train':
            loss.backward()
            optimizer.step()

            #network.eval()
            #update_lmbda_dual_val( args, logger, inputs, aug1, aug2, targets, network, temperature )
            #network.train()

        min_entropy_constraint.update( Ent_loss.item(), inputs.size(0) )
        algorithmic_stability_constraint.update( KD_loss.item(), inputs.size(0) )
        stable_constraint.update( Stable_loss.item(), inputs.size(0) )
        ft_constraint.update( Ft_loss.item(), inputs.size(0) )

        losses.update(loss.item(), inputs.size(0))
        constraint.update(_constraint.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if mode == 'train':
            scheduler.step(epoch)
            avg_scheduler.step(epoch)

            #if args.method == 'our-SD':
            #    parameter_copy( args, logger, teacher, network )
            #else:
            ema_update( args, logger, teacher, network )

        if (i % args.print_freq == 0) or (i == len(xloader)-1):
                #torch.cuda.empty_cache()
                progress.display(i)

    return losses.avg, top1.avg, top5.avg, min_entropy_constraint.avg, algorithmic_stability_constraint.avg, stable_constraint.avg, ft_constraint.avg



def get_updated_val(prev_budget, constraint):
          if prev_budget > constraint:
              budget_g = constraint / 2.
          else:    
              budget_g = constraint
          return budget_g    

def main(args):

    args.save_dir = args.save_dir + m__get_prefix( args ) + '--eval-' 
    print(args)

    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)

    criterion = nn.CrossEntropyLoss()

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )
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

    model = get_model_from_name( class_num, args.model_name, args.dataset )
    model = model.cuda()

    avg_model = get_model_from_name( class_num, args.model_name, args.dataset )
    avg_model = avg_model.cuda()

    if args._ckpt != "" and len(args._ckpt)>3:
        state = torch.load(args._ckpt)
        model.load_state_dict( state['base_state_dict'] )
        avg_model.load_state_dict( state['base_state_dict'] )

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

    avg_optimizer = torch.optim.SGD(avg_model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    avg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(avg_optimizer, args.epochs)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


    epoch=-1
    val_loss, val_acc1, val_acc5, min_entropy, algorithmic_stability, stable_const, ft_const = cifar_100_train_eval_loop( new_train_loader, args, logger, epoch, optimizer, scheduler, avg_optimizer, avg_scheduler, avg_model, model, 
            #train_loader, criterion, args.eval_batch_size, base=True, mode='eval' )
            valid_loader, criterion, args.eval_batch_size, base=True, mode='eval' )
    best_acc = val_acc1
    best_teacher_acc = val_acc1
    logger.log(' -- Best acc so far ' + str( best_acc ) + ' -- entropy ' + str( min_entropy ) + ' -- algorithmic stability ' + str( algorithmic_stability ) + ' -- stable ' + str(stable_const) + ' -- ft ' + str(ft_const) )




if __name__ == "__main__":

    parser = argparse.ArgumentParser( description="Self-Distillation CIFAR-10/100 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

        # args.budget_g_max 
        # args.budget_g_min 
        # args.constraint_val  
        # args.lmbda_increase
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

    parser.add_argument(
        "--_ckpt", type=str, 
        default='',
        help="The path to the model checkpoint"
    )

    parser.add_argument(
        "--KD_alpha", type=float, default=0.9, 
        help="The alpha parameter in knowledge distillation."
    )
    parser.add_argument(
        "--KD_temperature",
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

    parser.add_argument(
        "--model_name", type=str, 
        default='ResNet18',
        help="The path to the model configuration"
    )

    parser.add_argument("--method", type=str, default='CE', help="The method name. (CE, SD, our-SD).")
    parser.add_argument("--loss_type", type=str, default='CE', help="The method name. (CE, SD, our-SD).")

    # Data Generation
    parser.add_argument("--dataset", type=str, default='cifar10', help="The dataset name.")
    parser.add_argument("--data_path", type=str, default='/home/anilkag/code/compact-vision-nets-PDE-Feature-Generator/data/', help="The dataset name.")
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
        default=8,
        help="number of data loading workers (default: 8)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=2007, help="base model seed")
    parser.add_argument("--global_rand_seed", type=int, default=-1, help="global model seed")
    #add_shared_args(parser)

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
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00001,  help='weight decay')

    args = parser.parse_args()

    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"

    main(args)


