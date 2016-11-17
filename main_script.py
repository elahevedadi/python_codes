
#python libraries
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy
import math
import scipy
from scipy import signal
from sklearn import svm
import pdb
import random
import time

#my functions

from first_chain_functions import reduced_number_of_voxels
from first_chain_functions import make_train_and_test_data
from first_chain_functions import find_input_edges_regression_method_train
from first_chain_functions import find_test_cost
from first_chain_functions import find_mean_and_variance_of_theta
from first_chain_functions import test_train_check_func
from first_chain_functions import shift
from first_chain_functions import plotting_results 
############################################################################################33


example = nib.load("/home/elahe/Desktop/sub-01_ses-012_task-rest_run-001_bold.nii.gz")
data=example.get_data();

###################################################################

reduced_data = reduced_number_of_voxels(data)


num_train_examp = 0.7


x_train , x_test = make_train_and_test_data(reduced_data , num_train_examp )


t1 = x_train.shape[0]
n5 = x_train.shape[1]

rr_data = numpy.reshape(reduced_data,(n5,t1))

file_rr = open("rr_data.txt" , "w")
numpy.savetxt("rr_data.txt" , rr_data , fmt = '%.18e')


#############################################################################


# storing sets of theta_transpose for interpreting theta 


num_iter = 10000

alpha = 10

reduce_alpha_coef = 0.2

target_voxel_ind = 3382


num_storing_sets_of_theta = 5


         
my_theta = numpy.zeros(shape=(n5,num_storing_sets_of_theta))

my_cost_func = numpy.zeros(shape =(num_storing_sets_of_theta,1))

my_cost_func_per_iter = numpy.zeros(shape =(num_iter,num_storing_sets_of_theta))

my_test_cost = numpy.zeros(shape =(num_storing_sets_of_theta,1))

my_test_cost_per_iter = numpy.zeros(shape =(num_iter,num_storing_sets_of_theta))


for i in range(num_storing_sets_of_theta):
    theta_transpose, cost_func , cost_func_per_iter, test_cost , test_cost_per_iter = test_train_check_func(x_train , x_test ,
                                                                                                            target_voxel_ind ,
                                                                                                            alpha , num_iter ,
                                                                                                            reduce_alpha_coef)
    
    my_theta[:,i] = theta_transpose[:,0]
    my_cost_func[i] = cost_func
    my_cost_func_per_iter[:,i] = cost_func_per_iter
    my_test_cost[i] = test_cost
    my_test_cost_per_iter[:,i] = test_cost_per_iter




for i in range(num_storing_sets_of_theta):
    file1 = open("my_theta_"+str(i)+".txt" , "w")
    numpy.savetxt("my_theta_"+str(i)+".txt" , my_theta[:,i] , fmt = '%.18e')

    print("cost_func_"+str(i)+" "+"is "+str(my_cost_func[i]) , end='\n')
    print("test_cost_"+str(i)+" "+"is "+str(my_test_cost[i]) , end='\n')

    
    

#########################################################################

my_theta_mean , my_theta_variance = find_mean_and_variance_of_theta(my_theta)

file_mean = open("my_theta_mean.txt" , "w")
numpy.savetxt("my_theta_mean.txt" , my_theta_mean , fmt = '%.18e')

file_variance = open("my_theta_variance.txt" , "w")
numpy.savetxt("my_theta_variance.txt" , my_theta_variance , fmt = '%.18e')



####################################################################################333

test_cost = find_test_cost(x_test ,my_theta_mean , target_voxel_ind )
print("test_cost_theta_mean is "+str(test_cost) , end='\n')

######################################################################

plotting_results(my_cost_func_per_iter , my_test_cost_per_iter,
                     my_theta, 
                     my_theta_mean ,my_theta_variance,
                     num_storing_sets_of_theta)





    
    








     
