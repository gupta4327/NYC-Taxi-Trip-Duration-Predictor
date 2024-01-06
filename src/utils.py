from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def model_eval(model, x, y):

    '''This function takes model, input and output data and returns a scoring dictionary '''
    
    #predicting output from model
    y_pred = model.predict(x)

    #mean square error
    mse = round(mean_squared_error(y,y_pred),2)
        
    #root mean `square error
    rmse = round(np.sqrt(mean_squared_error(y,y_pred)),2)
    
    #mean absolute error
    mae = round(mean_absolute_error(y,y_pred),2)
    
    #root mean square percentage error
    rmspe = round(np.sqrt(np.sum(np.power(((y-y_pred)/y),2))/len(y)),3)
    
    #r2_score
    r2 = round(r2_score(y,y_pred),2)
    
    #adjusted_r2_score
    adjr2 = round(1-(1-r2_score(y,y_pred))*((x.shape[0]-1)/(x.shape[0]-x.shape[1]-1)),2)
    
    #dictionary storing all these testing score and this will be the returning value of function
    score_dict = {'Mean Square Error':mse, 'Root Mean Square Error':rmse,
                    'Mean Absolute Error':mae,'Root Mean Square Percentage Error':rmspe,'R2 Score':r2,
                    'Adjusted R2 Score': adjr2 }
    
    return score_dict