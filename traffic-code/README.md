
Below is a summary of the experiments performed in the traffic project:

- Multilayer Neural Networks:
    - Hidden Layer: tested several filter settings and didn't observe significant difference in accuracy
        - 4  units: loss: 0.2078 - accuracy: 0.9545
        - 24 units: loss: 0.2183 - accuracy: 0.9559
        - 16 units: loss: 0.1230 - accuracy: 0.9712
        
- Activation Functions
    - tested softmax and sigmoid in final output layer, accuracy difference is less than 0.01
        - softmax:  loss: 0.1328 - accuracy: 0.9690
        - sigmoid: loss: 0.1361 - accuracy: 0.9676

- Overfitting Avoidance
    - dropout
        - tested three dropput settings: 0.2, 0.5, and 0.9. Dropout 0.9's loss was 500% higher than dropout 0.2 and accuracy was 10% lower.
        - dropout 0.2: loss: 0.0056 - accuracy: 0.9837
        - dropout 0.5: loss: 0.0130 - accuracy: 0.9669
        - dropout 0.9: loss: 0.0258 - accuracy: 0.9068
    
- Image Convolution  
    - Max-Pooling: larger pool size accelarated the runtime slightly but decreased accuracy by 10%
        - (2, 2): 3s - loss: 0.1230 - accuracy: 0.9712
        - (8, 8): 2s - loss: 0.4130 - accuracy: 0.8998
        
- Standardization
    - Added batchnormalization layer for the 2-D convultion layer and dropout layer respectively, the normalization layer improved the process time and accuracy significantly. 

