# Neural Network 
###[Author: Hussain Mir Ali]
An artificial neural network I created with a single hidden layer. This project has been written in JavaScript. The applications include disease prediction, optimizing workout routine and stock prediction. 

npm link: https://www.npmjs.com/package/artificial-neural-network

To use the project:

run: npm install -g artificial-neural-network

and follow the sample usage provided below.

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
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1]
], [
    [1],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0]
]).then(console.log(nn.predict_result([[1,0,0,1,0,1]])));

/*Output
Training ...

Cost : 1.453592583906486
Cost : 0.4159937554245137
Cost : 0.025923046481012398
Cost : 0.009822261833421162
Cost : 0.0061083440270317405
Cost : 0.004465405304472341
Cost : 0.003535895517299884
Cost : 0.0029361733961879855
Cost : 0.002516116097524427
Matrix {
  _data: [ [ 0.03131379484990236 ] ],
  _size: [ 1, 1 ],
  _datatype: undefined }
*/
```
