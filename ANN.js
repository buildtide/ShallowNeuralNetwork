


"use strict";

var Neural_Network = function(args) {
    this.MathJS = require('mathjs');
    this.threshold = args.threshold||(1/this.MathJS.exp(6));
    this.iteration_callback = args.iteration_callback;
    this.q = require('q');
    this.regularization_param = args.regularization_param || 0.01; 
    this.learningRate = args.learningRate || 0.5;
    this.maximum_iterations = args.maximum_iterations || 0;
    this.notify_count = args.notify_count || 100;
};

Neural_Network.prototype.sigmoid = function(z) {
    var scope = {
            z: (typeof(z) === "number") ? this.MathJS.matrix([
                [z]
            ]) : z
        },
        sigmoid;

    scope.ones = this.MathJS.ones(scope.z.size()[0], scope.z.size()[1]);
    sigmoid = this.MathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))
    return sigmoid;
};

Neural_Network.prototype.forwardPropogation = function(X) {
    var y_result, X = this.MathJS.matrix(X) || this.x;
    this.z2 = this.MathJS.multiply(X, this.W1);
    this.a2 = this.sigmoid(this.z2);
    this.z3 = this.MathJS.multiply(this.a2, this.W2);
    y_result = this.sigmoid(this.z3);
    return y_result;
};

Neural_Network.prototype.sigmoid_Derivative = function(z) {
    var scope = {
            z: z
        },
        sigmoid_Derivative;
    scope.ones = this.MathJS.ones(z.size()[0], z.size()[1]);
    sigmoid_Derivative = this.MathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

    return sigmoid_Derivative;
};

Neural_Network.prototype.costFunction = function() {
    var J;
    var scope = {};
    this.y_result = this.forwardPropogation(this.x);
    scope.y_result = this.y_result;
    scope.y = this.y;
    scope.x = this.x;
    scope.W1 = this.W1;
    scope.W2 = this.W2;

    J = this.MathJS.sum(this.MathJS.eval('0.5*((y-y_result).^2)', scope))/(this.x.size()[0]) + (this.regularization_param/2)*this.MathJS.sum(this.MathJS.eval('W1.^2+W2.^2',scope));//regularization parameter

    return J;
};

Neural_Network.prototype.costFunction_Derivative = function() {
    this.y_result = this.forwardPropogation(this.x);
    var scope = {};
    scope.y_result = this.y_result;
    scope.y = this.y;
    scope.diff = this.MathJS.eval('-(y-y_result)', scope);
    scope.sigmoid_Derivative_z3 = this.sigmoid_Derivative(this.z3);
    scope.regularization_param = this.regularization_param;
    scope.W2 = this.W2;
    scope.W1 = this.W1;
    scope.m = this.x.size()[0];

    var del_3 = this.MathJS.eval('diff.*sigmoid_Derivative_z3', scope);
    var dJdW2 = this.MathJS.multiply(this.MathJS.transpose(this.a2), del_3);
        scope.dJdW2 = dJdW2;
        dJdW2 = this.MathJS.eval('dJdW2.*(1/m)',scope) + this.MathJS.eval('W2.*regularization_param', scope);

    scope.arrA = this.MathJS.multiply(del_3, this.MathJS.transpose(this.W2));
    scope.arrB = this.sigmoid_Derivative(this.z2);

    var del_2 = this.MathJS.eval('arrA.*arrB',scope);
    var dJdW1 = this.MathJS.multiply(this.MathJS.transpose(this.x), del_2);
        scope.dJdW1 = dJdW1;
        dJdW1 = this.MathJS.eval('dJdW1.*(1/m)',scope) + this.MathJS.eval('W1.*regularization_param', scope);


    return {'dJdW1': dJdW1, 'dJdW2': dJdW2};

};

Neural_Network.prototype.gradientDescent = function() {
    var gradient = new Array(2),
        cost,
        scope = {},
        defered = this.q.defer(),
        i = 0;
    console.log('Training ...\n');
    while (1) {
        gradient = this.costFunction_Derivative();
        scope.W1 = this.W1;
        scope.W2 = this.W2;
        scope.rate = this.learningRate;
        scope.dJdW1 = gradient.dJdW1;
        scope.dJdW2 = gradient.dJdW2;
        this.W2 = this.MathJS.eval('W2 - rate*dJdW2', scope);
        this.W1 = this.MathJS.eval('W1 - rate*dJdW1', scope);
        cost = this.costFunction();
        if (cost < (this.threshold)) {
            defered.resolve();
            break;
        }
        if (i % this.notify_count === 0 && iteration_callback !== undefined) {
             iteration_callback.apply(null, [cost, i/*iteration count*/]);//notify cost values for diagnosing the performance of learning algorithm.
        }
        i++;
        if(i>this.maximum_iterations)
        {
         defered.resolve();
         break;
        }
    }
    return defered.promise;
};

Neural_Network.prototype.train_network = function(X, Y) {
    this.x = this.MathJS.matrix(X);
    this.y = this.MathJS.matrix(Y);

    if ((this.y.size()[0] !== this.x.size()[0])) {
        console.log('\nPlease change the size of the input matrices so that X and Y have same number of rows.');
    }
   else {
        this.inputLayerSize = this.x.size()[1];
        this.outputLayerSize = 1;
        this.hiddenLayerSize = this.x.size()[1] + 1;
        this.W1 = (this.MathJS.random(this.MathJS.matrix([this.inputLayerSize, this.hiddenLayerSize]), -5, 5));
        this.W2 = (this.MathJS.random(this.MathJS.matrix([this.hiddenLayerSize, this.outputLayerSize*this.y.size()[1]]), -5, 5));
        }
        return this.gradientDescent();
};

Neural_Network.prototype.predict_result = function(X) {
    var y_result = this.forwardPropogation(X);
    return y_result;
};

module.exports = Neural_Network;

var nn = new Neural_Network(
    {'learning_rate':0.9, 
     'threshold_value':undefined /*optional threshold value*/, 
     'regularization_parameter': 0.01 /*optional regularization parameter to prevent overfitting*/, 
     'notify_count':  1000 /*optional value to execute the callback after every x number of iterations*/,
     'iteration_callback': undefined/*optional callback that can be used for getting cost and iteration value on every notify count.*/,
     'maximum_iterations': 100000  /*optional maximum iterations to be allowed*/});

nn.train_network([
    [1, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1]
], [
    [1,1,0],
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,0],
    [1,0,0],
    [1,1,0],
    [0,1,0]
]).then(console.log(nn.predict_result([[1,0,0,1,0,1]])),function(){},function(data){
    console.log(data);
});
