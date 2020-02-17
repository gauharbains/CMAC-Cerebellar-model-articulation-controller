'''
Discrete CMAC Neural Network
Author: Gauhar Bains
Graduate Robotics Student
University of Maryland, 
College Park, USA
'''

import math
import numpy as np
import random
import matplotlib.pyplot as plt


#create_dataset
min_dataset_range=0
max_dataset_range=2*math.pi
dataset_range=max_dataset_range-min_dataset_range
dataset=np.linspace(min_dataset_range,max_dataset_range,100).tolist()
dataset=[round(i,2) for i in dataset]

#random intialization of weights
weights= np.array(random.sample(range(0,100),35))/100
weights=np.ones_like(weights)

#split dataset in to test and train 70:30 and generate ground truth

#training data
test_data=[round(dataset[i],2) for i in range(4,len(dataset)-6,3)]
test_truth=[round(math.sin(i),2) for i in test_data ]
#test_truth=[round(math.pow(i,2),3) for i in test_data ]

# test data
training_data=[round(i,2) for i in dataset if i not in test_data]
training_truth=[round(math.sin(i),2) for i in training_data ]
#training_truth=[round(math.pow(i,2),3) for i in training_data ]

#normalized training and test input
norm_training_data=[i/dataset_range for i in training_data]
norm_test_data=[i/dataset_range for i in test_data]

#hyperparameters
#gen_factor=19
error_to_convergence=0.01
gen_factor=3
learning_rate=0.03
total_error=1000
max_iteration=500


while(abs(total_error)>error_to_convergence):
    
    total_error=0     
    for ind,i in enumerate(norm_training_data):
        central_weight=int(i*34)
        if gen_factor%2!=0 and gen_factor!=1 :
            lower_lim=central_weight-int(gen_factor/2) if central_weight-int(gen_factor/2)>=0 else 0
            upper_lim=central_weight+int(gen_factor/2)+1 if central_weight+int(gen_factor/2)+1<=34 else 34            
            asso_weights=[k for k in range(int(lower_lim),int(upper_lim))]     
            
        elif gen_factor==1:
            asso_weights=[central_weight]
        else:
            lower_lim=central_weight-int(gen_factor/2) if central_weight-int(gen_factor/2)>=0 else 0
            upper_lim=central_weight+int(gen_factor/2) if central_weight+int(gen_factor/2)<=34 else 34
            asso_weights=[k for k in range(lower_lim,upper_lim)]
        prediction=sum([weights[i] for i in asso_weights])
        error=training_truth[ind]-prediction
        total_error+=error
        error_per_weight=error/gen_factor
        for j in asso_weights:
            weights[j]=weights[j]+learning_rate*error_per_weight            
    
    

test_pred=[] 
for ind,i in enumerate(norm_test_data):
    central_weight=int(i*34)
    if gen_factor%2!=0:
        lower_lim=central_weight-int(gen_factor/2) if central_weight-int(gen_factor/2)>=0 else 0
        upper_lim=central_weight+int(gen_factor/2)+1 if central_weight+int(gen_factor/2)+1<=34 else 34
        asso_weights=[k for k in range(lower_lim,upper_lim)]
       
        
    else:
        lower_lim=central_weight-int(gen_factor/2) if central_weight-int(gen_factor/2)>=0 else 0
        upper_lim=central_weight+int(gen_factor/2) if central_weight+int(gen_factor/2)<=34 else 34
        asso_weights=[k for k in range(lower_lim,upper_lim)]
    prediction=sum([weights[i] for i in asso_weights])
    test_pred.append(round(prediction,2))

  
plt.plot(test_pred, label='Prediction')
plt.plot(test_truth, label='Test data')
plt.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.)
plt.show()
plt.savefig("Discreet_gen3.png")
    
    
        
    
    
    

    
    
        
        
        

    
    
    
        
        
        



