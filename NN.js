var Neural_Network = function(inputSize, hiddenSize, outPutSize, learningRate) {
    this.MathJS = require('mathjs');
    this.inputLayerSize = inputSize || 2;
    this.outputLayerSize = hiddenSize || 1;
    this.hiddenLayerSize = outPutSize || 3;
    this.learningRate = learningRate || 1;

    this.W1 = (this.MathJS.random(this.MathJS.matrix([this.inputLayerSize, this.hiddenLayerSize]), 5, 10));
    this.W2 = (this.MathJS.random(this.MathJS.matrix([this.hiddenLayerSize, this.outputLayerSize]), 5, 10));
};

Neural_Network.prototype.sigmoid = function(z) {
    var scope = {
        z: z
    };
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    scope.sigmoid = this.MathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))

    return scope.sigmoid;
};

Neural_Network.prototype.forwardPropogation = function(X) {
    var y_result;
    this.z2 = this.MathJS.multiply(X, this.W1);
    this.a2 = this.sigmoid(this.z2);
    this.z3 = this.MathJS.multiply(this.a2, this.W2)
    y_result = this.sigmoid(this.z3);
    return y_result;
};

Neural_Network.prototype.sigmoid_prime = function(z) {
    var scope = {
        z: z
    };
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    scope.sigmoid_prime = this.MathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

    return scope.sigmoid_prime;
};

Neural_Network.prototype.costFunction = function(X, Y) {
    var J;
    var scope = {};
    this.y_result = this.forward(X);
    scope.y_result = this.y_result;
    scope.y = Y;
    scope.x = x;

    J = this.MathJS.eval('0.5*((y-y_result).^2)', scope);

    return J;

};

Neural_Network.prototype.costFunction_Prime = function(X, Y) {
    this.y_result = this.forwardPropogation(X);
    var scope = {};
    scope.y_result = this.y_result;
    scope.y = Y;

    var del_3 = this.MathJS.multiply(this.MathJS.eval('-(y-y_result)', scope), this.sigmoid_prime(this.z3));

    var dJdW2 = this.MathJS.multiply(this.MathJS.transpose(this.a2), del_3);

    var del_2 = this.MathJS.multiply(this.MathJS.multiply(del_3, this.MathJS.transpose(this.W2)), this.sigmoid_prime(this.z2))

    var dJdW1 = this.MathJS.multiply(this.MathJS.transpose(a2), del_3);

    return [del_3, del_2];

};

Neural_Network.prototype.backPropogation = function() {
    //To be implemented.
};


var nn = new Neural_Network(undefined, undefined, undefined);
console.log(nn.forwardPropogation(nn.MathJS.matrix([
    [1, 5],
    [6, 6],
    [3, 4]
])));