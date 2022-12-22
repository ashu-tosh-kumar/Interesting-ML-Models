"""
To run the file, simply run the command python ruleof72.py from cmd in same location as this file

Machine Learning code to derive the Rule of 72
Rule of 72: "To estimate the length of time an amount of money takes to double,
simply divide its assumed annual growth rate into 72"
i.e. At 6% compounded annually, principal will be doubled in 72/CAGR = 72/6 = 12 years.

In this ML code, we aim to derive the value of 72 in formula 72/CAGR

Principal (P) is compounded annually at interest rate = InterestRate
After TimeYears years, amount = P(1+InterestRate/100)^TimeYears
Time taken for doubling the money:
Solve for TimeYears: P(1+InterestTate/100)^TimeYears = 2*P
Therefore, TimeToDouble = TimeYears = log(2)/log(1+InterestRate/100)
"""

import keras

# importing necessary libraries
import numpy as np
from keras import backend as K

# test dataset
# Assuming general interest rates vary from 1% to 40% annually
InterestRate = np.linspace(1, 40, 10000)  # x
InterestRate = InterestRate.reshape(InterestRate.shape[0], 1)
TimeToDouble = np.log(2) / np.log(1 + InterestRate / 100)  # y
TimeToDouble = TimeToDouble.reshape(TimeToDouble.shape[0], 1)


# keras model
# In this model, FinalLayerNode/InterestRate, estimates the TimeToDouble i.e.
# E(TimeToDouble) = FinalLayerNode/InterestRate and our aim is
# to estimate the E(FinalLayerNode/InterestRate) as according to the rule of 72
# Now, instead of comparing FinalLayerNode/InterestRate with TimeToDouble,
# we would instead compare FinalLayerNode with TimeToDouble*InterestRate
model_nn = keras.Sequential(
    [
        keras.layers.Dense(35, activation="relu", name="FirstLayer"),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(30, activation="relu", name="SecondLayer"),
        keras.layers.Dropout(0.01),
        keras.layers.Dense(1, activation="relu", name="FinalLayerNode"),
    ]
)

# compiling model_nn
model_nn.compile(optimizer="adam", loss="mean_squared_error")

# Training model_nn
history = model_nn.fit(InterestRate, InterestRate * TimeToDouble, epochs=500, batch_size=1000, verbose=True)

# model_nn summary
model_nn.summary()

# Computing node values for all layers of model_nn for all input InterestRate
outputs = [K.function([model_nn.input], [layer.output])([InterestRate, 1]) for layer in model_nn.layers]
FinalLayerNodeValuesArray = outputs[len(outputs) - 1][0]

# Required value in the rule of 72
# Note this value should exactly be 72 theoretically but we are getting value around 72
print(np.mean(FinalLayerNodeValuesArray))
