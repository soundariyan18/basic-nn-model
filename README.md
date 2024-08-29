# EX :1  Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

```
## PROGRAM
### Name: SOUNDARIYAN MN
### Register Number:212222230146
```

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_excel('dataa.xlsx')
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df

x=df[['Input']].values
y=df[['Output']].values
x

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scalar=MinMaxScaler()
scalar.fit(x_train)

x_train1=scalar.transform(x_train)
ai=Sequential([
    Dense (units = 8, activation = 'relu'),
    Dense (units = 10, activation = 'relu'),
    Dense (units = 1)])

ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(x_train1,y_train,epochs=2000)

loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

X_test1 = scalar.transform(x_test)
ai.evaluate(X_test1,y_test)

X_n1 = [[float(input('enter the value : '))]]
X_n1_1 = scalar.transform(X_n1)
a=ai.predict(X_n1_1)
print('The predicted output : ',a)
```


## Dataset Information
![image]("out.1.png")



## OUTPUT

### Training Loss Vs Iteration Plot
https://github.com/soundariyan18/basic-nn-model/blob/main/out.2.png

## Epoch
https://github.com/soundariyan18/basic-nn-model/blob/main/out.3.png


### Test Data Root Mean Squared Error
https://github.com/soundariyan18/basic-nn-model/blob/main/out.4.png



### New Sample Data Prediction
https://github.com/soundariyan18/basic-nn-model/blob/main/out.5.png



## RESULT

Thus a Neural network for Regression model is Implemented sucessfully.
