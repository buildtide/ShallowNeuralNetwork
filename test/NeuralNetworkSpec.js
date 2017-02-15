'use strict';
let NeuralNetwork = require('../NeuralNetwork');
let assert = require('assert');
let mathJS = require('mathjs');
let sinon = require('sinon');
let parse = require('csv-parse');

describe('NeuralNetwork', function() {

  let callback_data;

  global.localStorage = (function() {
    let storage = {};

    return {
      setItem: function(key, value) {
        storage[key] = value || '';
      },
      getItem: function(key) {
        return storage[key] || null;
      },
      removeItem: function(key) {
        delete storage[key];
      },
      get length() {
        return Object.keys(storage).length;
      },
      key: function(i) {
        let keys = Object.keys(storage);
        return keys[i] || null;
      }
    };
  })();

  let callback = function(data) {
    callback_data = data;
  };

  let nn = new NeuralNetwork({
    'hiddenLayerSize': 12,
    'learningRate': 0.9,
    'algorithm_mode': 0 ,
    'threshold_value': undefined /*optional threshold value*/ ,
    'regularization_parameter': 0.001 /*optional regularization parameter to prevent overfitting*/ ,
    'optimization_mode': {
      'mode': 0
    },
    'notify_count': 10 /*optional value to execute the callback after every x number of iterations*/ ,
    'iteration_callback': callback /*optional callback that can be used for getting cost and iteration value on every notify count.*/ ,
    'maximum_iterations': 500 /*optional maximum iterations to be allowed*/
  });

  let getInitParams = nn.getInitParams();

  it("should correctly set parameters", function() {
    assert.deepStrictEqual(getInitParams.learningRate, 0.9);
    assert.deepStrictEqual(getInitParams.hiddenLayerSize, 12);
    assert.deepStrictEqual(getInitParams.threshold, (1 / mathJS.exp(3)));
    assert.deepStrictEqual(getInitParams.regularization_param, 0.01);
    assert.deepStrictEqual(getInitParams.notify_count, 10);
    assert.deepStrictEqual(getInitParams.maximum_iterations, 500);
    assert.deepStrictEqual(getInitParams.iteration_callback, callback);
    assert.deepStrictEqual(getInitParams.optimization_mode, {
      'mode': 0
    });
  });

  describe('when saving and setting weights', function() {

    let W1 = (mathJS.random(mathJS.matrix([9, 10]), -5, 5)),
      W2 = (mathJS.random(mathJS.matrix([10, 3]), -5, 5));

    it('should successfuly save weights', function() {
      nn.saveWeights([W1, W2]);
      assert.deepStrictEqual([JSON.parse(global.localStorage.getItem("Weights"))[0].data, JSON.parse(global.localStorage.getItem("Weights"))[1].data], [W1._data, W2._data]);
    });

    it('should successfuly set weights', function(done) {
       assert.deepStrictEqual(nn.setWeights(),[W1._data, W2._data]);
       done();
    });
  });

  describe('when training', function() {

    let W1 = (mathJS.random(mathJS.matrix([10, 11]), -5, 5)),
      W2 = (mathJS.random(mathJS.matrix([11, 3]), -5, 5));
    let X = (mathJS.random(mathJS.matrix([10, 10]), 0, 1));
    let Y = (mathJS.random(mathJS.matrix([10, 3]), 0, 1));
    let y_result, z3, a2, z2;

    it("should correctly run sigmoid()", function() {
      let X = (mathJS.random(mathJS.matrix([10, 3]), 0, 1));
      let sigX = nn.sigmoid(X),
        i, j;
      let ones = mathJS.ones(X.size()[0], X.size()[1]);
      let success = false;
      let scope = {
        'z': X,
        'ones': ones
      };
      let refSigX = mathJS.eval('(ones+(e.^(z.*-1))).^-1', scope); //1/(1+e^(-z))

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
      let X = (mathJS.random(mathJS.matrix([10, 3]), 0, 1));
      let sigX = nn.sigmoid_Derivative(X),
        i, j;
      let ones = mathJS.ones(X.size()[0], X.size()[1]);
      let success = false;
      let scope = {
        'z': X,
        'ones': ones
      };
      let refSigX = mathJS.eval('(e.^(z.*-1))./(ones+(e.^(z.*-1))).^2', scope); //(1+e^(-z))/(1+e^(-z))^2

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
      
      let scope = {};
      y_result = nn.forwardPropagation(X, W1, W2, 1, 1);
      z2 = mathJS.multiply(X, W1);
      scope.z2 = z2;
      z2 = mathJS.eval('z2+1', scope);
      a2 = nn.sigmoid(z2);
      z3 = mathJS.multiply(a2, W2);
      scope.z3 = z3;
      z3 = mathJS.eval('z3+1', scope);
      let y_resultRef = nn.sigmoid(z3);
      let i, j;
      let success = false;

      for (i = 0; i < y_resultRef.size()[0]; i++) {
        for (j = 0; j < y_resultRef.size()[1]; j++) {
          if (y_result._data[i][j] !== y_resultRef._data[i][j]) {
            success = false;
            break;
          } else {
            success = true;
          }
          if (j == y_resultRef.size()[1])
            j = 0;
        }
      }

      assert.equal(success, true);

    });

    it("should correctly run costFunction()", function() {

      let scope = {
        'y_result': y_result,
        'y': Y,
        'x': X,
        'W1': W1,
        'W2': W2
      };
      let success = true;

      let J1 = mathJS.sum(mathJS.eval('0.5*((y-y_result).^2)', scope)) / (scope.x.size()[0]) + (getInitParams.regularization_param / 2) * (mathJS.sum(mathJS.eval('W1.^2', scope)) + mathJS.sum(mathJS.eval('W2.^2', scope))); //regularization parameter

          nn.setBias(1, 1);

      let cost1 = nn.costFunction(X, Y, 0);
     
      if (cost1 !== J1)
        success = false;

      assert.equal(success, true);

    });

    it("should correctly run costFunction_Derivative()", function() {
      let scope = {};
      scope.y_result = y_result;
      scope.y = Y;
      scope.diff = mathJS.eval('-(y-y_result)', scope);
      scope.sigmoid_Derivative_z3 = nn.sigmoid_Derivative(z3);
      scope.regularization_param = getInitParams.regularization_param;
      scope.W2 = W2;
      scope.W1 = W1;
      scope.m = X.size()[0];
      let success = false;

      let del_3 = mathJS.eval('diff.*sigmoid_Derivative_z3', scope);
      let dJdW2 = mathJS.multiply(mathJS.transpose(a2), del_3);
      scope.dJdW2 = dJdW2;
      scope.regularization_term_dJdW2 = mathJS.eval('W2.*regularization_param', scope);
      dJdW2 = mathJS.eval('dJdW2.*(1/m) + regularization_term_dJdW2', scope);

      scope.arrA = mathJS.multiply(del_3, mathJS.transpose(W2));
      scope.arrB = nn.sigmoid_Derivative(z2);

      let del_2 = mathJS.eval('arrA.*arrB', scope);
      let dJdW1 = mathJS.multiply(mathJS.transpose(X), del_2);
      scope.dJdW1 = dJdW1;
      scope.regularization_term_dJdW1 = mathJS.eval('W1.*regularization_param', scope);
      dJdW1 = mathJS.eval('dJdW1.*(1/m) + regularization_term_dJdW1', scope);

      let dJdWRef = nn.costFunction_Derivative(X, Y, W1, W2);
      let i, j;

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
      let success = true;
      let spy = sinon.spy(nn, "saveWeights");
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
      let success = true;
      let spy = sinon.spy(nn, "costFunction_Derivative");
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
      let success = false;

      nn.gradientDescent(X, Y, W1, W2).then(function(data) {
        if (data[0] === callback_data.cost && (data[1] - 1) === callback_data.iteration) {
          success = true;

        }

        assert.equal(success, true);
        done();
      });
    });

    it("should call costFunction() while running gradientDescent()", function(done) {
      let success = true;
      let spy = sinon.spy(nn, "costFunction");
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
      let success = false;

      nn.gradientDescent(X, Y, W1, W2).then(function(data) {

        if ((data[0] <= (1 / mathJS.exp(3))) || (data[1] - 1) === getInitParams.maximum_iterations)
          success = true;
        assert.equal(success, true);
        done();
      });
    });

    describe('when predicting the result', function() {

      it("should call setWeights()", function() {
        let spy = sinon.spy(nn, "setWeights");
        nn.predict_result(X);
        spy.restore();
        assert.deepStrictEqual(spy.calledOnce, true);
      });

      it("should call forward_Propagation()", function() {
        let spy = sinon.spy(nn, "forwardPropagation");
        nn.predict_result(X);
        spy.restore();
        assert.deepStrictEqual(spy.calledOnce, true);
      });
    });
  });
});
