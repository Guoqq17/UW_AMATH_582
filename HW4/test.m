load fisheriris;
tree=fitctree(meas,species,'OptimizeHyperparameters','auto');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree)