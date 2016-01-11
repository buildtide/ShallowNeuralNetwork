# Advance Neural Network 
###[Author: Hussain Mir Ali]
An artificial neural network with single hidden layer and multiclass classification. This project has been written in JavaScript. The applications include disease prediction, optimizing workout routine and stock prediction. 

##External Librarbies Used:
* csv-parse License: https://github.com/wdavidw/node-csv-parse/blob/master/LICENSE
* mathjs License: https://github.com/josdejong/mathjs/blob/master/LICENSE
* mocha License: https://github.com/mochajs/mocha/blob/master/LICENSE
* sinon Licencse: https://github.com/sinonjs/sinon/blob/master/LICENSE
* yuidocjs License: https://github.com/yui/yuidoc/blob/master/LICENSE
* nodeJS License: https://github.com/nodejs/node/blob/master/LICENSE

##Note: 
* Please perform Feature Scaling and/or Mean Normalization along with random shuffling of data for using this program.

##Installation:
*  Download the project and unzip it.
*  Copy the 'advance-neural-network' folder to your node_modules folder in your project directory.
*  Require it using 'require('advance-neural-network')' in your main JavaScript file.
*  If you want to reinstall node_modules for this project then run 'sudo npm install -g" in your terminal under the 'advance-neural-network' project directory.

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
var NeuralNetwork = require('advance-neural-network');
var callback_data;

var callback = function (data) {
    console.log(data);
    callback_data = data;
};

var nn = new NeuralNetwork({
     'path': undefined, /*optional path to save the weights*/
     'learningRate':0.9, 
     'hiddenLayerSize': 9,
     'threshold_value':undefined /*optional threshold value*/, 
     'regularization_parameter': 0.001 /*optional regularization parameter to prevent overfitting*/, 
     'notify_count':  100/*optional value to execute the callback after every x number of iterations*/,
     'iteration_callback': callback/*optional callback that can be used for getting cost and iteration value on every notify count.*/,
     'maximum_iterations': 10000 /*optional maximum iterations to be allowed*/});

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
    [0, 1, 0, 0, 1, 0]],[
    [1,1,1],
    [1,0,1]]
    ))).then(
console.log(nn.test_network([
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0]],
    [[0,1,1],
    [1,0,0]]))).then(console.log(nn.predict_result([[1,0,0,1,0,1]])));  

*/
```
