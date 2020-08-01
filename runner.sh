#!/usr/bin/env bash
# Shell script for creating all result directories and guiding over running the commands 
mkdir -p output_reports/{aruba,dsads,hapt,hatn6,milan,opp,pamap,twor,ws,mnist,permuted_mnist}
mkdir -p vis_outputs/{accuracy_vis/{acc_by_batches,acc_by_classes,detail_acc},corr_vis/{by_predictions,by_raw_features},exemp_vis,norm_vis/{by_batches,by_epochs},per_person,tsne_vis}

python runner.py --dataset 'hapt' --total_classes 12 --new_classes 2 --base_classes 2 --epochs 200 --method 'kd_kldiv_wa1' # for incremental learning
python runner.py --dataset 'hapt' --total_classes 12 --base_classes 12 --new_classes 0 --epochs 200 --method 'kd_kldiv_wa1' # for offline learning in single batch
python runner.py --dataset 'dsads' --total_classes 19 --base_classes 2 --new_classes 2 --epochs 200 --method 'kd_kldiv_wa1' --exemplar 'icarl' # e.g. for ICaRL-styled exemplar selection

python runner.py --dataset 'permuted_mnist' --total_classes 10 --new_classes 10 --base_classes 10 --epochs 5 --method 'agem' # for verification on permuted-mnist
python runner.py --dataset 'mnist' --total_classes 10 --new_classes 2 --base_classes 2 --epochs 5 --method 'agem' # for verification on split-mnist
