# NeuralNetwork 
###[Author: Hussain Mir Ali]
An artificial neural network with single hidden layer written in JavaScript programming language. The applications include disease prediction, optimizing workout routine and stock prediction. In the NN.js file I have used sample data.

To run the project:

clone the project in your local repo.

run 'sudo npm install -g'

run 'node NN.js'

###Sample usage:

```javascript
var Neural_Network = require('artificial-neural-network');
var nn = new Neural_Network();

nn.train_network(0.9, [
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1]
], [
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1]
]).then(console.log(nn.predict_result([[1,0,1,0,1,1]])));
