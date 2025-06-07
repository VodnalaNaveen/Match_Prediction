# importing the requrired libraries

import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import time


if __name__ == "__main__":

    # creating the object to pass the arguments
    parser = argparse.ArgumentParser()

    # adding argument 1 to the script
    parser.add_argument("--weights_path",type=str,required=True,help="pass the weights path")
    
    # adding argument 2 to the script
    parser.add_argument("--data_path",type=str,required=False,default="./data.csv",help = "pass the data path")
    
    # adding argument 3 to the script
    parser.add_argument('--num_preds',type=int,required=True,help='how many predictions you need ?')
    
    # declaring the arguments
    args = parser.parse_args()

    weights_path = args.weights_path # adding weights path to the variable
    data_path = args.data_path # adding data path to the variable
    num_preds = args.num_preds # adding number of predictions to the num_preds 


    # Loading the model
    model = tensorflow.keras.models.load_model(weights_path)
    
    # Loading the data
    data = pd.read_csv(data_path)

    # Dropping the unwanted columns
    data.drop(columns=["Timestamp","Email Address"],axis=1,inplace=True)
    
    # Encoding the Data
    for column in data.columns:
        le = le.fit(data[column])
        data[column] = le.fit_transform(data[column])
    # print(data.shape)
    
    c = 0
    timetaken = 0
    predicted_labels = []
    for row in range(len(data)):
        if c == num_preds:
            break
        observation = data.loc[row]   # Extracting each row
        observation = np.array(observation) # converting each row into numpy array
        observation = observation.reshape(1,10) # reshaping the row into (1,10)
        start = time.time()
        output=model.predict(observation) # making the prediction
        end = time.time()
        total = end - start # calculating the time
        timetaken += total

        if output[0][0] < 0.5:
                predicted_labels.append(0)
        else:
                predicted_labels.append(1)
        c +=1
        
    # calculating the total time for num_preds provided
    print(f"total time taken for {num_preds} prediction is {timetaken:.4f} seconds")
    for i in predicted_labels:
        if i == 1:
            print("💘 It's a Match!")
        else:
            print("💔 No Match.")
    
    
    
    
    
  
    
    