
##################
def reduced_number_of_voxels(input_data):
     
        import pdb
        import random
        import time
        import os
        import numpy
        import math
        import scipy
     
        x = input_data.shape[0] 
        y = input_data.shape[1]
        z = input_data.shape[2]
        t = input_data.shape[3]
        n = x*y*z
        reduced_data = numpy.zeros(shape=(x//5,y//5,z//5,t))
        for tt in range(t):
    
            for m in range(x//5):
                for n in range(y//5):
                    for p in range(z//5):
                        i = (5*(m+1))-3
                        j = (5*(n+1))-3
                        k = (5*(p+1))-3
                        totaldata = [input_data[i-2,j+2,k,tt],input_data[i-1,j+2,k,tt],input_data[i,j+2,k,tt],input_data[i+1,j+2,k,tt],input_data[i+2,j+2,k,tt],
                                     input_data[i-2,j+1,k,tt],input_data[i-1,j+1,k,tt],input_data[i,j+1,k,tt],input_data[i+1,j+1,k,tt],input_data[i+2,j+1,k,tt],
                                     input_data[i-2,j,k,tt],input_data[i-1,j,k,tt],input_data[i,j,k,tt],input_data[i+1,j,k,tt],input_data[i+2,j,k,tt],
                                     input_data[i-2,j-1,k,tt],input_data[i-1,j-1,k,tt],input_data[i,j-1,k,tt],input_data[i+1,j-1,k,tt],input_data[i+2,j-1,k,tt],
                                     input_data[i-2,j-2,k,tt],input_data[i-1,j-2,k,tt],input_data[i,j-2,k,tt],input_data[i+1,j-2,k,tt],input_data[i+2,j-2,k,tt],
                                     input_data[i-2,j+2,k-2,tt],input_data[i-1,j+2,k-2,tt],input_data[i,j+2,k-2,tt],input_data[i+1,j+2,k-2,tt],input_data[i+2,j+2,k-2,tt],
                                     input_data[i-2,j+1,k-2,tt],input_data[i-1,j+1,k-2,tt],input_data[i,j+1,k-2,tt],input_data[i+1,j+1,k-2,tt],input_data[i+2,j+1,k-2,tt],
                                     input_data[i-2,j,k-2,tt],input_data[i-1,j,k-2,tt],input_data[i,j,k-2,tt],input_data[i+1,j,k-2,tt],input_data[i+2,j,k-2,tt],
                                     input_data[i-2,j-1,k-2,tt],input_data[i-1,j-1,k-2,tt],input_data[i,j-1,k-2,tt],input_data[i+1,j-1,k-2,tt],input_data[i+2,j-1,k-2,tt],
                                     input_data[i-2,j-2,k-2,tt],input_data[i-1,j-2,k-2,tt],input_data[i,j-2,k-2,tt],input_data[i+1,j-2,k-2,tt],input_data[i+2,j-2,k-2,tt],
                                     input_data[i-2,j+2,k-1,tt],input_data[i-1,j+2,k-1,tt],input_data[i,j+2,k-1,tt],input_data[i+1,j+2,k-1,tt],input_data[i+2,j+2,k-1,tt],
                                     input_data[i-2,j+1,k-1,tt],input_data[i-1,j+1,k-1,tt],input_data[i,j+1,k-1,tt],input_data[i+1,j+1,k-1,tt],input_data[i+2,j+1,k-1,tt],
                                     input_data[i-2,j,k-1,tt],input_data[i-1,j,k-1,tt],input_data[i,j,k-1,tt],input_data[i+1,j,k-1,tt],input_data[i+1,j,k-1,tt],
                                     input_data[i-2,j-1,k-1,tt],input_data[i-1,j-1,k-1,tt],input_data[i,j-1,k-1,tt],input_data[i+1,j-1,k-1,tt],input_data[i+2,j-1,k-1,tt],
                                     input_data[i-2,j-2,k-1,tt],input_data[i-1,j-2,k-1,tt],input_data[i,j-2,k-1,tt],input_data[i+1,j-2,k-1,tt],input_data[i+2,j-2,k-1,tt],
                                     input_data[i-2,j+2,k+1,tt],input_data[i-1,j+2,k+1,tt],input_data[i,j+2,k+1,tt],input_data[i+1,j+2,k+1,tt],input_data[i+2,j+2,k+1,tt],
                                     input_data[i-2,j+1,k+1,tt],input_data[i-1,j+1,k+1,tt],input_data[i,j+1,k+1,tt],input_data[i+1,j+1,k+1,tt],input_data[i+2,j+1,k+1,tt],
                                     input_data[i-2,j,k+1,tt],input_data[i-1,j,k+1,tt],input_data[i,j,k+1,tt],input_data[i+1,j,k+1,tt],input_data[i+1,j,k+1,tt],
                                     input_data[i-2,j-1,k+1,tt],input_data[i-1,j-1,k+1,tt],input_data[i,j-1,k+1,tt],input_data[i+1,j-1,k+1,tt],input_data[i+2,j-1,k+1,tt],
                                     input_data[i-2,j-2,k+1,tt],input_data[i-1,j-2,k+1,tt],input_data[i,j-2,k+1,tt],input_data[i+1,j-2,k+1,tt],input_data[i+2,j-2,k+1,tt],
                                     input_data[i-2,j+2,k+2,tt],input_data[i-1,j+2,k+2,tt],input_data[i,j+2,k+2,tt],input_data[i+1,j+2,k+2,tt],input_data[i+2,j+2,k+2,tt],
                                     input_data[i-2,j+1,k+2,tt],input_data[i-1,j+1,k+2,tt],input_data[i,j+1,k+2,tt],input_data[i+1,j+1,k+2,tt],input_data[i+2,j+1,k+2,tt],
                                     input_data[i-2,j,k+2,tt],input_data[i-1,j,k+2,tt],input_data[i,j,k+2,tt],input_data[i+1,j,k+2,tt],input_data[i+1,j,k+2,tt],
                                     input_data[i-2,j-1,k+2,tt],input_data[i-1,j-1,k+2,tt],input_data[i,j-1,k+2,tt],input_data[i+1,j-1,k+2,tt],input_data[i+2,j-1,k+2,tt],
                                     input_data[i-2,j-2,k+2,tt],input_data[i-1,j-2,k+2,tt],input_data[i,j-2,k+2,tt],input_data[i+1,j-2,k+2,tt],input_data[i+2,j-2,k+2,tt]]

                                                          
                                                                                 
                        reduced_data[m,n,p,tt]= numpy.mean(totaldata)
        return reduced_data                
    
#################################################


def make_train_and_test_data(input_reduced_data , num_train_examp ):
    
    import pdb
    import random
    import time
    import os
    import numpy
    import math
    import scipy

    x1 = input_reduced_data.shape[0] 
    y1 = input_reduced_data.shape[1]
    z1 = input_reduced_data.shape[2]
    t1 = input_reduced_data.shape[3]
    n5 = x1*y1*z1

    rr_data_transpose = numpy.transpose(numpy.reshape(input_reduced_data,(n5,t1)))
    
    x_train = rr_data_transpose[0:int((num_train_examp)*t1) , :]

    x_test  = rr_data_transpose[int((num_train_examp*t1)) : t1 , :]

    return x_train , x_test


##########################################################
def shift(xs, n):
        import pdb
        import random
        import time
        import os
        import numpy
        import math
        import scipy
        if n >= 0:
           return numpy.r_[np.full(n, 0), xs[:-n]]
        else:
           return numpy.r_[xs[-n:], numpy.full(-n, 0)]
#########################################################################        

# regression_function training

def find_input_edges_regression_method_train( input_x_train , target_voxel_ind , alpha , num_iter ,reduce_alpha_coef):
                                        
     # p is number of target voxel and it is in range[0:4693]
     # alpha is steps in gradient descend
     # num_iter is number of gradient descend iteration
     
     import pdb
     import random
     import time
     import os
     import numpy
     import math
     import scipy
     
     t1 = input_x_train.shape[0] 
     n5 = input_x_train.shape[1]
     
    
     
     theta_transpose = numpy.random.seed(int(100 * time.clock()))
     theta_transpose = numpy.random.random((n5 , 1 )) #initial theta whith random matrix
     theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
     
     
     
     
     x_train_normalized = numpy.zeros(shape=(t1 , n5))
    

     x_train_normalized = (input_x_train)/(0.0001 + numpy.linalg.norm(input_x_train))

     
     train_label_normalized = x_train_normalized[:,target_voxel_ind ]
     
     
     #gradient descend algorithm
     cost_func_per_iter = numpy.zeros(shape=(num_iter))
     s = numpy.zeros(shape = (n5,1))
     
     
            
     
     for ite in range(num_iter):
             cost_func = 0
              
             for i in range(t1  - 1):
                
                     
                 hypo_func = numpy.dot((x_train_normalized[i,:]),(theta_transpose))
                     
                 temp= (( hypo_func- train_label_normalized[i+1] ) * x_train_normalized[i,:])# i and i+1 is because of causality
                 s = s+numpy.reshape(temp,[n5,1])
                 if i% (t1 - 1) == 0:
                    theta_transpose =  theta_transpose - (alpha/(reduce_alpha_coef*ite+1)) * (2/(t1/2)) *s
                    theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
                    s = numpy.zeros(shape = (n5,1))
                    cost_func =  cost_func + (2/t1) * ( math.pow(( hypo_func - train_label_normalized[i+1] ) , 2))

                        
                                
             theta_transpose[target_voxel_ind ] = 0 # so we remove train_label from x_train      
             cost_func_per_iter[ite] = cost_func
             
              
                   
     return theta_transpose, cost_func , cost_func_per_iter

##############################################################################################

#regression_function test

def find_test_cost(input_x_test , input_theta_transpose , target_voxel_ind ):
    import pdb
    import random
    import time
    import os
    import numpy
    import math
    import scipy

    
    n5  = input_x_test.shape[1]
    t1 = input_x_test.shape[0]
  
    test_label = input_x_test[:,target_voxel_ind ]
    test_cost = 0

    x_test_normalized = numpy.zeros(shape=(t1 , n5))

    x_test_normalized = (input_x_test)/(0.0001 + numpy.linalg.norm(input_x_test))

        
     
    hypo_func = numpy.dot((x_test_normalized) , (input_theta_transpose)) # it is a m*1 or (t1/2 * 1) matrix
    
    test_label_normalized = x_test_normalized[:,target_voxel_ind ]

   


     
    test_cost =  (1/t1) * math.pow((numpy.linalg.norm( hypo_func - shift(test_label_normalized , -1))) , 2)
    
    


    return test_cost 

############################################################################################################

    

def find_mean_and_variance_of_theta(input_my_theta):

    import pdb
    import random
    import time
    import os
    import numpy
    import math
    import scipy
    
    my_theta_mean = numpy.mean(input_my_theta , axis=1)

    my_theta_mean = (my_theta_mean)/(0.0001 + numpy.linalg.norm(my_theta_mean))
    
    my_theta_variance = numpy.var(input_my_theta , axis = 1)

    return my_theta_mean , my_theta_variance

###############################################################################################################################

def test_train_check_func( input_x_train , input_x_test , target_voxel_ind , alpha , num_iter ,reduce_alpha_coef):
                                        
     # p is number of target voxel and it is in range[0:4693]
     # alpha is steps in gradient descend
     # num_iter is number of gradient descend iteration
     
     import pdb
     import random
     import time
     import os
     import numpy
     import math
     import scipy
     
     t1 = input_x_train.shape[0] 
     n5 = input_x_train.shape[1]
     
    
     
     theta_transpose = numpy.random.seed(int(100 * time.clock()))
     theta_transpose = numpy.random.random((n5 , 1 )) #initial theta whith random matrix
     theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
     
     
     
     
     x_train_normalized = numpy.zeros(shape=(t1 , n5))

    
     x_train_normalized = (input_x_train)/(0.0001 + numpy.linalg.norm(input_x_train))

     train_label_normalized = x_train_normalized[:,target_voxel_ind ]

     n5  = input_x_test.shape[1]
     t1 = input_x_test.shape[0]
  
     test_label = input_x_test[:,target_voxel_ind ]
     test_cost = 0

     x_test_normalized = numpy.zeros(shape=(t1 , n5))

     x_test_normalized = (input_x_test)/(0.0001 + numpy.linalg.norm(input_x_test))
     
     
     #gradient descend algorithm
     cost_func_per_iter = numpy.zeros(shape=(num_iter))
     s = numpy.zeros(shape = (n5,1))
     test_cost_per_iter = numpy.zeros(shape=(num_iter))
     
     
            
     
     for ite in range(num_iter):
             cost_func = 0
              
             for i in range(t1  - 1):
                
                     
                 hypo_func = numpy.dot((x_train_normalized[i,:]),(theta_transpose))
                     
                 temp= (( hypo_func- train_label_normalized[i+1] ) * x_train_normalized[i,:])# i and i+1 is because of causality
                 s = s+numpy.reshape(temp,[n5,1])
                 if i% (t1 - 1) == 0:
                    theta_transpose =  theta_transpose - (alpha/(reduce_alpha_coef*ite+1)) * (2/(t1/2)) *s
                    theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
                    s = numpy.zeros(shape = (n5,1))
                    cost_func =  cost_func + (2/t1) * ( math.pow(( hypo_func - train_label_normalized[i+1] ) , 2))

                        
                                
             theta_transpose[target_voxel_ind ] = 0 # so we remove train_label from x_train      
             cost_func_per_iter[ite] = cost_func
             
             hypo_func = numpy.dot((x_test_normalized) , (theta_transpose)) # it is a m*1 or (t1/2 * 1) matrix
    
             test_label_normalized = x_test_normalized[:,target_voxel_ind ]

             test_cost =  (1/t1) * math.pow((numpy.linalg.norm( hypo_func - shift(test_label_normalized , -1))) , 2)

             test_cost_per_iter[ite] = test_cost  
             
              
                   
     return theta_transpose, cost_func , cost_func_per_iter, test_cost , test_cost_per_iter

######################################################3

def plotting_results(input_my_cost_func_per_iter , input_my_test_cost_per_iter,
                     input_my_theta, 
                     input_my_theta_mean ,input_my_theta_variance,
                     num_storing_sets_of_theta):
        import numpy
        import math
        import matplotlib.pyplot as plt
        for i in range(num_storing_sets_of_theta):
                
                plt.figure(1)
                plt.plot(numpy.log10(input_my_cost_func_per_iter[:,i]))
                plt.title("logaritm plot of "+str(num_storing_sets_of_theta)+
                          "cost_func_per_iter with the same parameters")
                plt.xlabel("number of iterations")
                plt.ylabel("cost_func_per_iter")

                

                plt.figure(2)
                plt.plot(numpy.log10(input_my_test_cost_per_iter[:,i]))
                plt.title("logaritm plot of "+str(num_storing_sets_of_theta)+
                          "test_cost_per_iter with the same parameters")
                plt.xlabel("number of iterations")
                plt.ylabel("test_cost_per_iter")



                plt.figure(3)
                plt.plot(input_my_theta[:,i])
                plt.title(" plot of "+str(num_storing_sets_of_theta)+
                          "theta with the same parameters for all voxels")
                plt.xlabel("number of voxels")
                plt.ylabel("theta")



                plt.figure(4)
                plt.plot(input_my_theta[1000:1020,i])
                plt.title(" plot of "+str(num_storing_sets_of_theta)+
                          "theta with the same parameters for  voxel 1000 to 1020")
                plt.xlabel("voxel 1000 to 1020")
                plt.ylabel("theta")


                

        plt.figure(5)
        plt.plot(input_my_theta_mean)
        plt.title(" plot of theta_mean")
        plt.xlabel("number of voxels")
        plt.ylabel("theta_mean")




        plt.figure(6)
        plt.plot(input_my_theta_variance)
        plt.title(" plot of theta_variance")
        plt.xlabel("number of voxels")
        plt.ylabel("theta_variance")




        


        



                


                
                

                
                     

        








