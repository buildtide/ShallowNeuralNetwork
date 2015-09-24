var Neural_Network = function(inputSize, hiddenSize, outPutSize, learningRate, X, Y) {
    this.MathJS = require('mathjs');
    this.x = this.MathJS.matrix(X);
    this.y = this.MathJS.matrix(Y);

    this.inputLayerSize = inputSize || 2;
    this.outputLayerSize = hiddenSize || 1;
    this.hiddenLayerSize = outPutSize || 3;
    this.learningRate = learningRate || 1;

    this.W1 = (this.MathJS.random(this.MathJS.matrix([this.inputLayerSize, this.hiddenLayerSize]), -5, 5));
    this.W2 = (this.MathJS.random(this.MathJS.matrix([this.hiddenLayerSize, this.outputLayerSize]), -5, 5));
};

Neural_Network.prototype.sigmoid = function(z) {
    var scope = {
        z: z
    }, sigmoid;
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    sigmoid = this.MathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))

    return sigmoid;
};

Neural_Network.prototype.forwardPropogation = function(X) {
    var y_result, X = X || this.x;
    this.z2 = this.MathJS.multiply(X, this.W1);
    this.a2 = this.sigmoid(this.z2);
    this.z3 = this.MathJS.multiply(this.a2, this.W2)
    y_result = this.sigmoid(this.z3);
    return y_result;
};

Neural_Network.prototype.sigmoid_prime = function(z) {
    var scope = {
        z: z
    }, sigmoid_prime;
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    sigmoid_prime = this.MathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

    return sigmoid_prime;
};

Neural_Network.prototype.costFunction = function() {
    var J;
    var scope = {};
    this.y_result = this.forwardPropogation(this.x);
    scope.y_result = this.y_result;
    scope.y = this.y;
    scope.x = this.x;

    J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope));

    return J;
};

Neural_Network.prototype.costFunction_Prime = function() {
    this.y_result = this.forwardPropogation(this.x);
    var scope = {};
    scope.y_result = this.y_result;
    scope.y = this.y;
    scope.diff = this.MathJS.eval('-(y-y_result)', scope);
    scope.sigmoid_prime_z3 = this.sigmoid_prime(this.z3);

    var del_3 = this.MathJS.eval('diff.*sigmoid_prime_z3',scope);

    var dJdW2 = this.MathJS.multiply(this.MathJS.transpose(this.a2), del_3);

    var del_2 = this.MathJS.multiply(this.MathJS.multiply(del_3, this.MathJS.transpose(this.W2)), this.sigmoid_prime(this.z2))

    var dJdW1 = this.MathJS.multiply(this.MathJS.transpose(this.x), del_2);

    return [dJdW1, dJdW2];

};

Neural_Network.prototype.gradientDescent = function() {
    var gradient = new Array(2), scope = {}, i = 0;
    console.log('Training\n');
    while(1) {
	   gradient = this.costFunction_Prime();
       scope.W1 = this.W1;
       scope.W2 = this.W2;
       scope.rate = this.learningRate;
       scope.dJdW1 = gradient[0];
       scope.dJdW2 = gradient[1];
       this.W2 = this.MathJS.eval('W2 - rate*dJdW2', scope); 
       this.W1 = this.MathJS.eval('W1 - rate*dJdW1', scope);
	   cost = this.costFunction() 
       if(cost < (1/this.MathJS.exp(6)))
		 break;
       if(i%100 === 0)
       {
        console.log('Cost : '+cost);
       }
    }
};

Neural_Network.prototype.predict = function(X) {
    return this.forwardPropogation(X);
};

var nn = new Neural_Network(undefined, undefined, undefined, 0.5, [
    [1, 5],
    [6, 6],
    [3, 4]
], [
    [1],
    [6],
    [3]
]);

    nn.gradientDescent();
