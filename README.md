# Shallow Neural Network 
###[Author: Hussain Mir Ali]
An artificial neural network with single hidden layer and multiclass classification. This project has been written in JavaScript. The applications include disease prediction, optimizing workout routine and stock prediction. 

##External Librarbies Used:
* mathjs License: https://github.com/josdejong/mathjs/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE
* q License: https://github.com/kriskowal/q/blob/v1/LICENSE

##Note: 
* Please perform Feature Scaling and/or Mean Normalization along with random shuffling of data for using this program.

##Installation:
*  Download the project and unzip it.
*  Copy the 'NeuralNetwork' folder to your project directory.


##Testing:
* For unit testing Mocha and Sinon have been used. 
* On newer computers run the command 'mocha --timeout 50000', the 50000 ms timeout is to give enough time for tests to complete as they might not process before timeout. 
* On older computers run the command 'mocha --timeout 300000', the 300000 ms timeout is to give enough time for tests to complete as they might not process before timeout on older computers. 
* If need be more than 300000 ms should be used to run the tests depending on the processing power of the computer. 

##Documentation
*  The documentation is available in the 'out' folder of this project. Open the 'index.html' file under the 'out' folder with Crhome or Firefox.
*  To generate the documentation run 'yuidoc .' command in the main directory of this project.

###Sample usage:

```javascript
//main.js file
var callback_data;

var callback = function (data) {
    console.log(data);
    callback_data = data;
};

var nn =  new window.NeuralNetwork({
        'hiddenLayerSize': 12,
        'learningRate': 0.1,
        'algorithm_mode': 0 /*This is to specify if  training:0, cross validating:1 or testing:2 data.*/ ,
        'threshold': undefined /*optional threshold value*/ ,
        'regularization_parameter': 0.001 /*optional regularization parameter to prevent overfitting*/ ,
        'optimization_mode': {
          'mode': 1,
          'batch_size': 2
        } /*optional optimization mode for type of gradient descent. {mode:1, 'batch_size': <your size>} for mini-batch and {mode: 0} for batch.*/ ,
        'notify_count': 10 /*optional value to execute the callback after every x number of iterations*/ ,
        'iteration_callback': callback /*optional callback that can be used for getting cost and iteration value on every notify count.*/ ,
        'maximum_iterations': 100 /*optional maximum iterations to be allowed*/
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
]).then(
console.log(nn.cross_validate_network([   
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0]],[
    [1,1,1],
    [1,0,1],
    [1,1,1],
    [1,0,1]]
    ))).then(
console.log(nn.test_network([
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0]],[
    [1,1,1],
    [1,0,1],
    [1,1,1],
    [1,0,1]]))).then(console.log(nn.predict_result([[1,0,0,1,0,1]])));  

*/
```
```
<!--index.html-->
<!doctype html>
<html>
  <head>
  </head>
  <body >
        <script src="neuralnetwork/lib/q.js"></script>
        <script src="neuralnetwork/lib/math.js"></script>
        <script src="neuralnetwork/NeuralNetwork.js"></script>
         <!--Include the main.js file where you use the algorithm.-->
        <script src="main.js"></script>
</body>
</html>

*/
```