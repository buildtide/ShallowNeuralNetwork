# Shallow Neural Network 
### [Author: Hussain Mir Ali]
#### [Hire me,](mailto:hkmirali+npm@gmail.com) I can build your next project. 

<img src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Artificial_Neural_Network.jpg"  width="500px" alt="shallownetwork"></img>

An artificial neural network with single hidden layer and multiclass classification. The applications include modelling non-linear data.

## Note: 
* Please perform Feature Scaling and/or Mean Normalization along with random shuffling of data for using this program.

## Installation:
*  Run 'npm install @softnami/neuralnetwork'.

### Sample usage:

```javascript
//main.js file

import NeuralNetwork from '@softnami/neuralnetwork';

let callback_data;

//callback for diagnostics
let callback = function (data) {
    console.log(data);
    callback_data = data;
};

let nn =  new NeuralNetwork({
        'hiddenLayerSize': 12,
        'learningRate': 0.1,
        'threshold': undefined /*optional threshold value for cost. Defaults to 1/(e^3).*/ ,
        'regularization_parameter': 0.001 /*optional regularization parameter to prevent overfitting. Defaults to 0.01.*/ ,
        'optimization_mode': {
          'mode': 1,
          'batch_size': 2
        } /*optional optimization mode for type of gradient descent. {mode:1, 'batch_size': <your size>} for mini-batch and {mode: 0} for batch. Defaults to batch gradient descent.*/ ,
        'notify_count': 10 /*optional value to execute the callback after every x number of iterations. Defaults to 100. */ ,
        'iteration_callback': callback /*optional callback that can be used for getting cost and iteration value on every notify count. Defaults to empty function.*/ ,
        'maximum_iterations': 100 /*optional maximum iterations to be allowed. Defaults to 1000.*/
      });

nn.train_network([
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1]
], [
    [1,1,1],
    [1,0,1],
    [0,1,0],
    [1,0,0],
    [1,1,0],
    [0,1,0]
]).then(console.log(nn.predict_result([[1,0,0,1,0,1]])));  

*/
```

## Testing:
* For unit testing Mocha and Sinon have been used. 
* Run 'npm test', if timeout occurs then increase timeout in test script.

## Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the  documentation run 'yuidoc .' command in the main directory of this project.

