var NeuralNetwork = require('../NeuralNetwork');
var assert = require('assert');
var mathJS = require('mathjs');
var sinon = require('sinon');
var parse = require('csv-parse');

describe('NeuralNetwork', function() {

	var callback_data, path = new Array("./test/Test_Weights_Layer1.txt", "./test/Test_Weights_Layer2.txt");

	var callback = function(data) {
		callback_data = data;
	};

	var nn = new NeuralNetwork({
		'path': path,
		/*optional path to save the weights*/
		'learningRate': 0.9,
		'algorithm_mode': 0 /*This is to specify if  testing:0, cross validating:1 or training:2 data.*/ ,
		'threshold_value': undefined /*optional threshold value*/ ,
		'regularization_parameter': 0.001 /*optional regularization parameter to prevent overfitting*/ ,
		'notify_count': 10 /*optional value to execute the callback after every x number of iterations*/ ,
		'iteration_callback': callback /*optional callback that can be used for getting cost and iteration value on every notify count.*/ ,
		'maximum_iterations': 1000 /*optional maximum iterations to be allowed*/
	});

	var getInitParams = nn.getInitParams();

	it("should correctly set parameters", function() {
		assert.deepStrictEqual(getInitParams.path[0], './test/Test_Weights_Layer1.txt');
		assert.deepStrictEqual(getInitParams.path[1], './test/Test_Weights_Layer2.txt');
		assert.deepStrictEqual(getInitParams.learningRate, 0.9);
		assert.deepStrictEqual(getInitParams.algorithm_mode, 0);
		assert.deepStrictEqual(getInitParams.threshold, (1 / mathJS.exp(6)));
		assert.deepStrictEqual(getInitParams.regularization_param, 0.01);
		assert.deepStrictEqual(getInitParams.notify_count, 10);
		assert.deepStrictEqual(getInitParams.maximum_iterations, 1000);
		assert.deepStrictEqual(getInitParams.iteration_callback, callback);
	});

	describe('when saving and setting weights', function() {

		var W1 = (mathJS.random(mathJS.matrix([9, 10]), -5, 5)),
			W2 = (mathJS.random(mathJS.matrix([10, 3]), -5, 5));

		it('should successfuly save weights', function() {
			assert.equal(nn.saveWeights([W1, W2], path), true);
		});

		it('should successfuly set weights', function(done) {
			var success = true;
			nn.setWeights(path).then(function(promise_data) {
				assert(promise_data.success, true);
				done();
			});

		});

	});

	describe('when training', function() {

		var W1 = (mathJS.random(mathJS.matrix([10, 11]), -5, 5)),
			W2 = (mathJS.random(mathJS.matrix([11, 3]), -5, 5));
		var X = (mathJS.random(mathJS.matrix([10, 10]), 0, 1));
		var Y = (mathJS.random(mathJS.matrix([10, 3]), 0, 1));
		var y_result, z3, a2, z2;

		it("should correctly run sigmoid()", function() {
			var X = (mathJS.random(mathJS.matrix([10, 3]), 0, 1));
			var sigX = nn.sigmoid(X),
				i, j;
			var ones = mathJS.ones(X.size()[0], X.size()[1]);
			var success = false;
			var scope = {
				'z': X,
				'ones': ones
			};
			var refSigX = mathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))

			for (i = 0; i < sigX.size()[0]; i++) {
				for (j = 0; j < sigX.size()[1]; j++) {
					if (sigX._data[i][j] !== refSigX._data[i][j]) {
						success = false;
						break;
					} else
						success = true;

					if (j == sigX.size()[1])
						j = 0;
				}
			}
			assert.equal(success, true);

		});

		it("should correctly run sigmoid_Derivative()", function() {
			var X = (mathJS.random(mathJS.matrix([10, 3]), 0, 1));
			var sigX = nn.sigmoid_Derivative(X),
				i, j;
			var ones = mathJS.ones(X.size()[0], X.size()[1]);
			var success = false;
			var scope = {
				'z': X,
				'ones': ones
			};
			var refSigX = mathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

			for (i = 0; i < sigX.size()[0]; i++) {
				for (j = 0; j < sigX.size()[1]; j++) {
					if (sigX._data[i][j] !== refSigX._data[i][j]) {
						success = false;
						break;
					} else
						success = true;

					if (j == sigX.size()[1])
						j = 0;
				}
			}
			assert.equal(success, true);
		});

		it("should correctly run forwardPropagation()", function() {

			y_result = nn.forwardPropagation(X, W1, W2);
			z2 = mathJS.multiply(X, W1);
			a2 = nn.sigmoid(z2);
			z3 = mathJS.multiply(a2, W2);
			var y_resultRef = nn.sigmoid(z3);
			var i, j;
			var success = false;

			for (i = 0; i < y_resultRef.size()[0]; i++) {
				for (j = 0; j < y_resultRef.size()[1]; j++) {
					if (y_result._data[i][j] !== y_resultRef._data[i][j]) {
						success = false;
						break;
					} else
						success = true;

					if (j == y_resultRef.size()[1])
						j = 0;
				}
			}

			assert.equal(success, true);

		});

		it("should correctly run costFunction()", function() {

			var scope = {
				'y_result': y_result,
				'y': Y,
				'x': X,
				'W1': W1,
				'W2': W2
			};
			var success = true;

			var J1 = mathJS.sum(mathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]),
				J2 = mathJS.sum(mathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]) + (getInitParams.regularization_param / 2) * (mathJS.sum(mathJS.eval('W1.^2', scope)) + mathJS.sum(mathJS.eval('W2.^2', scope))); //regularization parameter

			var cost1 = nn.costFunction(X, Y, 0);
			var cost2 = nn.costFunction(X, Y, 1);
			var cost3 = nn.costFunction(X, Y, 2);

			if (cost1 !== J2)
				success = false;
			if (cost2 !== J1 || cost3 !== J1)
				success = false;
			if (cost2 !== cost3)
				success = false;

			assert.equal(success, true);

		});

		it("should correctly run costFunction_Derivative()", function() {
			var scope = {};
			scope.y_result = y_result;
			scope.y = Y;
			scope.diff = mathJS.eval('-(y-y_result)', scope);
			scope.sigmoid_Derivative_z3 = nn.sigmoid_Derivative(z3);
			scope.regularization_param = getInitParams.regularization_param;
			scope.W2 = W2;
			scope.W1 = W1;
			scope.m = X.size()[0];
			var success = false;

			var del_3 = mathJS.eval('diff.*sigmoid_Derivative_z3', scope);
			var dJdW2 = mathJS.multiply(mathJS.transpose(a2), del_3);
			scope.dJdW2 = dJdW2;
			scope.regularization_term_dJdW2 = mathJS.eval('W2.*regularization_param', scope);
			dJdW2 = mathJS.eval('dJdW2.*(1/m) + regularization_term_dJdW2', scope);

			scope.arrA = mathJS.multiply(del_3, mathJS.transpose(W2));
			scope.arrB = nn.sigmoid_Derivative(z2);

			var del_2 = mathJS.eval('arrA.*arrB', scope);
			var dJdW1 = mathJS.multiply(mathJS.transpose(X), del_2);
			scope.dJdW1 = dJdW1;
			scope.regularization_term_dJdW1 = mathJS.eval('W1.*regularization_param', scope);
			dJdW1 = mathJS.eval('dJdW1.*(1/m) + regularization_term_dJdW1', scope);

			var dJdWRef = nn.costFunction_Derivative(X, Y, W1, W2);
			var i, j;

			for (i = 0; i < dJdW1.size()[0]; i++) {
				for (j = 0; j < dJdW1.size()[1]; j++) {
					if (dJdWRef[0]._data[i][j] !== dJdW1._data[i][j]) {
						success = false;
						break;
					} else
						success = true;

					if (j == dJdW1.size()[1])
						j = 0;
				}
			}

			for (i = 0; i < dJdW2.size()[0]; i++) {
				for (j = 0; j < dJdW2.size()[1]; j++) {
					if (dJdWRef[1]._data[i][j] !== dJdW2._data[i][j]) {
						success = false;
						break;
					} else
						success = true;

					if (j == dJdW2.size()[1])
						j = 0;
				}
			}

			assert.equal(success, true);

		});

		it("should call saveWeights() while running gradientDescent()", function(done) {
			var success = true;
			var spy = sinon.spy(nn, "saveWeights");
			spy.withArgs([W1, W2]);

			nn.gradientDescent(X, Y, W1, W2).then(function(data) {

				if (!spy.called)
					success = false;
				spy.restore();
				assert.equal(success, true);
				done();

			});

		});

		it("should call costFunction_Derivative() while running gradientDescent()", function(done) {
			var success = true;
			var spy = sinon.spy(nn, "costFunction_Derivative");
			spy.withArgs(X, Y, W1, W2);

			nn.gradientDescent(X, Y, W1, W2).then(function(data) {

				if (!spy.called)
					success = false;
				spy.restore();
				assert.equal(success, true);
				done();

			});

		});

		it("should call iteration_callback() while running gradientDescent()", function(done) {
			var success = false;

			nn.gradientDescent(X, Y, W1, W2).then(function(data) {
				if (data[0] === callback_data.cost && (data[1] - 1) === callback_data.iteration) {
					success = true;

				}

				assert.equal(success, true);
				done();
			});
		});

		it("should call costFunction() while running gradientDescent()", function(done) {
			var success = true;
			var spy = sinon.spy(nn, "costFunction");
			spy.withArgs(X, Y);

			nn.gradientDescent(X, Y, W1, W2).then(function(data) {
				if (!spy.called)
					success = false;
				spy.restore();
				assert.equal(success, true);
				done();

			});
		});

		it("should correctly run gradientDescent()", function(done) {
			var success = false;

			nn.gradientDescent(X, Y, W1, W2).then(function(data) {

				if ((data[0] <= (1 / mathJS.exp(6))) || (data[1] - 1) === getInitParams.maximum_iterations)
					success = true;
				assert.equal(success, true);
				done();
			});
		});

		describe('when predicting the result', function() {

			it("should call setWeights()", function() {
				var spy = sinon.spy(nn, "setWeights");
				spy.withArgs(X);
				nn.predict_result(X);
				spy.restore();
				assert.deepStrictEqual(spy.calledOnce, true);
			});

			it("should call forward_Propagation()", function() {
				var spy = sinon.spy(nn, "forwardPropagation");
				spy.withArgs(X);
				nn.predict_result(X);
				spy.restore();
				assert.deepStrictEqual(spy.calledOnce, true);
			});
		});

		describe('when testing', function() {

			it("should call costFunction() and set algorithm_mode to 2", function() {
				var success = true;
				var spy = sinon.spy(nn, "costFunction");
				spy.withArgs(X, Y, undefined);
				var result = nn.test_network(X, Y);
				var params = nn.getInitParams();

				if (params.algorithm_mode !== 2)
					success = false;
				if (!spy.calledOnce)
					success = false;
				spy.restore();

				assert.deepStrictEqual(success, true);
			});
		});

		describe('when cross validating', function() {

			it("should call costFunction() and set algorithm_mode to 1", function() {
				var success = true;
				var spy = sinon.spy(nn, "costFunction");
				spy.withArgs(X, Y, undefined);
				var result = nn.cross_validate_network(X, Y);
				var params = nn.getInitParams();

				if (params.algorithm_mode !== 1)
					success = false;
				if (!spy.calledOnce)
					success = false;
				spy.restore();
				assert.deepStrictEqual(success, true);

			});
		});

	});
});