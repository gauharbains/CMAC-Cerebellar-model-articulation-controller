# CMAC-Cerebellar-model-articulation-controller

I’ve programmed a discrete and continuous CMAC consisting of 35 weights using python. It is trained to predict f(x)= sin(x). A dataset of 100 evenly spaced points was constructed between the range (0,2π). Following this, the dataset was split into test and train in the ratio of 30:70.
The result of best perfomring discreet(generalization factor = 5) CMAC is shown below :
![Discreet_gen5](https://user-images.githubusercontent.com/48079888/75101571-6f675400-55ac-11ea-8694-25842150c27f.png)

## Results
The effect of overlap area on accuracy and time to convergence was analysed. 
The performance of CMAC was analyzed for generalization factor ranging between 1 to 34. To evaluate the performance of the network, two parameters have been used:
- MAPE : Mean Absolute Percentage Error
- RMSE: Root Mean Square Error

 For more information, please read the report attached.
