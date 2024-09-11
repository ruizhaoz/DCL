import csv 
import matplotlib.pyplot as plt

#filename = 'debug_info/aug-cifar100-our-SD-ResNet18-KLS-MinEnt-Ft9-200-5-200-10.0-0.05-0.05-0.05-0.05-0.3-2.0-1.0-debug----debug-info.csv'
#filename = 'debug_info/aug-cifar100-our-SD-ResNet18-KLS-MinEnt-Ft9-200-5-200-10.0-0.05-0.05-0.05-0.05-0.3-2.0-1.0-debug----debug-info-2.csv'
#filename = 'debug_info/aug-cifar100-our-SD-ResNet18-KLS-MinEnt-Ft9-200-5-200-10.0-0.05-0.05-0.05-0.05-0.3-2.0-1.0-debug----debug-info-3.csv'
filename = 'debug_info/aug-cifar100-our-SD-ResNet18-KLS-MinEnt-Ft9-200-5-200-10.0-0.05-0.05-0.05-0.05-0.3-2.0-1.0-debug----debug-info-4.csv'

with open(filename, mode='r') as fp:
    csvfile = csv.reader(fp)
    header = next(csvfile)

    trn_robust_list = []
    tst_robust_list = []
    budget_robust_list = []

    trn_stable_list = []
    tst_stable_list = []
    budget_stable_list = []

    for lines in csvfile:
        stage, epoch, trn_robust, tst_robust, trn_stable, tst_stable = int(lines[0]), int(lines[1]), float(lines[2]), float(lines[3]), float(lines[4]), float(lines[5]) 

        budget_g, budget_g_stable = float(lines[6]), float(lines[7])
        only_trn_acc, trn_acc, val_acc = float(lines[8]),float(lines[9]),float(lines[10])
        only_trn_ce, trn_ce, val_ce = float(lines[11]),float(lines[12]),float(lines[13])

        budget_robust_list.append(budget_g)
        budget_stable_list.append(budget_g_stable)

        trn_robust_list.append( trn_robust )
        tst_robust_list.append( tst_robust )

        trn_stable_list.append( trn_stable )
        tst_stable_list.append( tst_stable )

o_trn_stable_list = trn_stable_list
o_tst_stable_list = tst_stable_list
o_budget_stable_list = budget_stable_list

iters = 300

trn_robust_list = trn_robust_list[:iters]
tst_robust_list = tst_robust_list[:iters]
budget_robust_list = budget_robust_list[:iters]

trn_stable_list = trn_stable_list[:iters]
tst_stable_list = tst_stable_list[:iters]
budget_stable_list = budget_stable_list[:iters]

fig = plt.figure()
filename = './figs/robustness_trn_vs_tst_plot.png'

plt.plot( trn_robust_list, label='Train' )
plt.plot( tst_robust_list, label='Test' )
#plt.plot( budget_robust_list, label='Budget' )

plt.legend( loc='upper right' )
plt.xlabel('Train Iterations')
plt.ylabel('Robustness')

plt.savefig(filename, bbox_inches='tight', dpi=600)
plt.close( fig )


fig = plt.figure()
filename = './figs/stability_trn_vs_tst_plot.png'

plt.plot( trn_stable_list, label='Train' )
plt.plot( tst_stable_list, label='Test' )
#plt.plot( budget_stable_list, label='Budget' )

plt.legend( loc='upper right' )
plt.xlabel('Train Iterations')
plt.ylabel('Stability')

plt.savefig(filename, bbox_inches='tight', dpi=600)
plt.close( fig )




iters = 600

trn_stable_list = o_trn_stable_list[:iters]
tst_stable_list = o_tst_stable_list[:iters]

budget = 0.05
o_budget_stable_list = [ budget ] * 200 + [budget/2] * 200 + [budget/4] * 200 
budget_stable_list = o_budget_stable_list[:iters]

fig = plt.figure()
filename = './figs/stability_plot_constraint_violations.png'

plt.plot( trn_stable_list, label='Train Constraint Val' )
plt.plot( budget_stable_list, label='Target Budget' )

plt.legend( loc='upper right' )
plt.xlabel('Train Iterations')
plt.ylabel('Stability')

plt.savefig(filename, bbox_inches='tight', dpi=600)
plt.close( fig )
