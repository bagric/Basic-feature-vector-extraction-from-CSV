import EVM, numpy, scipy
evm = EVM.MultipleEVM(tailsize=0)

class1 = numpy.random.normal((0,0),3,(50,2))
class2 = numpy.random.normal((-10,10),3,(50,2))
class3 = numpy.random.normal((10,-10),3,(50,2))
print(class1)
print(class2)
print(class3)

evm.train([class1, class2, class3])
probability, evm_index = evm.max_probabilities([[-8,8], [13, -14]])
print(probability)
print(evm_index)
print(evm_index[1][0])