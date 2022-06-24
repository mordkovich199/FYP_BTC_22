####BENCHMARKING###

### >> Specifications
#-test set sizes: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#-R^2 can't be evaluated if test data is shuffled, but RMSE can, therefore
#shuffle will be used to find the best RMSE, but R^2 calculated afterwards
*Not so sure about the above statement.


#for the best set of parameters without shuffle.
#-R^2 only focuses on variance while RMS makes sure that S2F line gets close to the actual $ price
###

### >> Results
#
#I. (Lin) 
#Average Test RMSE: 8345.27$
#Average Test R^2: 0.54
#Worst RMSE: 9272.74$
#Worst R^2: 0.14
#
#II. (Lin + z-score)
#_in z_ 
#Average Test RMSE: 8857.72$
#Average Test R^2: 0.85
#Worst RMSE: 9151.96$
#Worst R^2: 0.42
#_in $_
#Average Test RMSE: 1569.94$
#Average Test R^2: 0.85
#Worst RMSE: 1897.30$
#Worst R^2: 0.42
#
#III. (Lin + PCA)
#Average Test RMSE: 10613.65$
#Average Test R^2: 0.54
#Worst RMSE: 12123.61$
#Worst R^2: 0.13
#
#VI. (Lin + PCA + z-score)
#_in z_
#Average Test RMSE: 9781.77$
#Average Test R^2: 0.72
#Worst RMSE: 10161.99$
#Worst R^2: 0.38
#_in $_
#Average Test RMSE: 2493.02$
#Average Test R^2: 0.72
#Worst RMSE: 2847.58$
#Worst R^2: 0.38
#
#V. (NN)
#Average Test RMSE:
#Average Test R^2:
#
#VI. (NN + z-score)
#Average Test RMSE:
#Average Test R^2:
#
#VII. (NN + PCA)
#Average Test RMSE:
#Average Test R^2:
#
#VIII (NN + PCA + z-score)
#Average Test RMSE:
#Average Test R^2:

#- - - - - - - - - - - - - -

#(S2F) [s2f = f1 * sf^f2]
#Best Train Params:
#Average Test RMSE:
#Average Test R^2:




###
#II. (Lin + z-score)
#_in z_ 
#Average Test RMSE: 8857.72$
#Average Test R^2: 0.85
#Worst RMSE: 9151.96$
#Worst R^2: 0.42

#VI. (Lin + PCA + z-score)
#_in z_
#Average Test RMSE: 9781.77$
#Average Test R^2: 0.72
#Worst RMSE: 10161.99$
#Worst R^2: 0.38

PCA (n=4)
RMSE values: 5000
Average RMSE: 2387.0433502810843
Best RMSE: 2074.6743660725547
Worst RMSE: 2709.6077000978025
Average R^2: 0.7732347251464106
Best R^2: 0.8822948001286572
Worst R^2: 0.690312837278829






--------Ugly Learnign Curve--------
Average Validaion RMSE: 3979
Average Validation R^2: 0.8062909874509626

Average Test  RMSE: 5410
Average Test R^2: 0.8988617230769584

--------Better Learning Curve--------
Average Validaion RMSE: 4077
Average Validation R^2: 0.8065543684532948

Average Test  RMSE: 4729
Average Test R^2: 0.8743361152324021

Outcome: much better RMSE but worse R^2
