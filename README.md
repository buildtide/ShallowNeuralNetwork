# NeuralNetwork 
###[Author: Hussain Mir Ali]
An artificial neural network with single hidden layer written in JavaScript programming language. The applications include disease prediction, optimizing workout routine and stock prediction. 

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

/*Output
Training ...

Cost : 1.8672996948914875
Cost : 1.499937311080165
Cost : 1.4963081436834709
Cost : 1.416919492518602
Cost : 0.1755899386143785
Cost : 0.06734046733229879
Cost : 0.03572883581799631
Cost : 0.021715024540958933
Cost : 0.015229381553306442
Cost : 0.011728920684120863
Cost : 0.00964741962254102
Cost : 0.008341141315080585
Cost : 0.00747495077635342
Cost : 0.006843143303305748
Cost : 0.006332478563678655
Cost : 0.005894389794841658
Cost : 0.005510116542077858
Cost : 0.005170836871139338
Cost : 0.004870607655329581
Cost : 0.0046044300492293685
Cost : 0.00436784585163636
Cost : 0.00415689298525935
Cost : 0.003968102050073072
Cost : 0.003798471938862676
Cost : 0.0036454264860645595
Cost : 0.0035067633173530024
Cost : 0.003380602921108016
Cost : 0.0032653419615453574
Cost : 0.0031596122337993983
Cost : 0.003062245308800236
Cost : 0.0029722423496271544
Cost : 0.002888748422585989
Cost : 0.002811030647769474
Cost : 0.00273845961680057
Cost : 0.002670493596107223
Cost : 0.002606665112736969
Cost : 0.0025465695825218114
Cost : 0.002489855689574085
Matrix {
  _data: [ [ 0.9989169660012883 ] ],
  _size: [ 1, 1 ],
  _datatype: undefined }

*/
```
