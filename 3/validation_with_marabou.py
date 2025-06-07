import numpy as np

from maraboupy import Marabou

options = Marabou.createOptions(verbosity = 0)

filename = "my_model.onnx"
network = Marabou.read_onnx(filename)

# input and output variable numbers; [0] since first dimension is batch size
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

# Set input bounds
network.setLowerBound(inputVars[0], 0.0)
network.setUpperBound(inputVars[0], 1.0)
network.setLowerBound(inputVars[1], 0.0)
network.setUpperBound(inputVars[1], 40.0)
network.setLowerBound(inputVars[2], 20.0)
network.setUpperBound(inputVars[2], 40.0)

# Set output bounds
network.setLowerBound(outputVars[0], 4.0+(1e-5))
network.setUpperBound(outputVars[0], -(1e-5))

# Call to Marabou solver
exitCode, vals, stats = network.solve(options = options)
print(exitCode, vals, stats)

# Test Marabou equations against onnxruntime at an example input point
inputPoint = np.ones(inputVars.shape)

network = Marabou.read_onnx(filename)
marabouEval = network.evaluateWithMarabou([inputPoint], options = options)[0]
onnxEval = network.evaluateWithoutMarabou([inputPoint])[0]

# The two evaluations should produce the same result
print("\n############\n")
print("Marabou Evaluation:")
print(marabouEval)
print("\nONNX Evaluation:")
print(onnxEval)
print("\nDifference:")
print(onnxEval - marabouEval)
assert max(abs(onnxEval - marabouEval).flatten()) < 1e-6