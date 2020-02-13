# Multilayer-Perceptron

Training of MLP for XOR Implementation with 4 examples, 2 inputs, 2hidden nodes, 1 Output node and predicting output

A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to refer to any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation); see § Terminology. Multilayer perceptrons are sometimes colloquially referred to as "vanilla" neural networks, especially when they have a single hidden layer.

An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable[source : Wikipedia]

# Getting Started
There are two files in the repository
 1. 18200300_Achin_Tamiri_program.py
 
 2. requirements.txt
 
 #Installation
 
 1 . Clone the master branch onto you computer using 
 
 git clone https://github.com/achintamiri/Multilayer-Perceptron.git
 
 2 . pip install requirements.txt
 
 # Execution
 python 18200300_Achin_Tamiri_program.py
 
 # Sample Results
 
####Inputs per node : 2 Output per nodes:  1 Total Examples 4
 
####First Input Layer weights

 [[0.22610361 0.31747275]
 [0.41350185 0.82768737]
 [0.03715283 0.84418771]]

 ####Second hidden Layer weights:
 
 [[0.15971483]
 [0.50496708]
 [0.36814746]]
 
####Final Optimized weights obtained at epoch 6236

####Best Optimized weights for hidden layer

 [[-2.94913902]
 [-7.08066765]
 [ 6.59867944]]
 
 ####Best Optimized weights for input  layer
 
 [[-4.93965186 -2.14266023]
 [ 3.25120436  5.50649927]
 [ 3.24328573  5.46086669]]

####XOR Results :

| Inputs | Predicted Output | Threshold Round Off Output Set(0.5) |   |   |
|--------|------------------|-------------------------------------|---|---|
| [0 0]  | [0.09058865]     | 0                                   |   |   |
| [0 1]  | [0.91066831]     | 1                                   |   |   |
| [1 0]  | [0.91087817]     | 1                                   |   |   |
| [1 1]  | [0.09999232]     | 0                                   |   |   |

####Inputs per node :  4 Output per node : 1 Total Training Examples : 150

####First Input Layer weights

 [[0.03798149 0.53173104 0.4949744  0.81307856 0.0692259 ]
 [0.02864592 0.84250294 0.32078802 0.85604693 0.37416741]
 [0.30472335 0.04337829 0.07352161 0.51542925 0.6833361 ]
 [0.30316658 0.29648333 0.4093306  0.58786459 0.54717666]
 [0.31789933 0.3785422  0.72244325 0.84319335 0.42054736]]
 
 ####Second hidden Layer weights:
 
 [[0.29585048]
 [0.43623687]
 [0.31662961]
 [0.4668164 ]
 [0.15539275]
 [0.02629546]]
 
*****Model Training for input Sin Examples (150)***

####Final Optimized weights obtained at epoch 1

####Predict final y_train
 
 [[0.00424802]
 [0.00237187]
 [0.00297125]
 [0.0029175 ]
 [0.00255312]
 [0.00258218]
 [0.00470597]
 [0.00297038]
 [0.00507801]
 [0.00256273]
 [0.00391149]
 [0.00314025]
 [0.00215249]
 [0.00216424]
 [0.0032177 ]
 [0.00274121]
 [0.00280953]
 [0.00497247]
 [0.00345894]
 [0.00236145]
 [0.00267141]
 [0.00360855]
 [0.00314021]
 [0.00284286]
 [0.00333243]
 [0.00350869]
 [0.00197446]
 [0.00240309]
 [0.00349811]
 [0.00572424]
 [0.00261991]
 [0.00568192]
 [0.00349041]
 [0.00314685]
 [0.00407664]
 [0.00427733]
 [0.0021366 ]
 [0.00247116]
 [0.00303215]
 [0.00244128]
 [0.00242944]
 [0.00429045]
 [0.00248132]
 [0.00261458]
 [0.00264442]
 [0.00236926]
 [0.00281422]
 [0.00266154]
 [0.00344187]
 [0.00385315]
 [0.00297074]
 [0.00285261]
 [0.0038766 ]
 [0.00406004]
 [0.00412897]
 [0.00343278]
 [0.00244701]
 [0.00475889]
 [0.00261197]
 [0.00232012]
 [0.00233589]
 [0.00272817]
 [0.00218743]
 [0.00327802]
 [0.00203378]
 [0.00324883]
 [0.00231199]
 [0.00234391]
 [0.00285626]
 [0.00249497]
 [0.00199415]
 [0.00230782]
 [0.0027999 ]
 [0.00364809]
 [0.00256468]
 [0.00470527]
 [0.00443242]
 [0.00299544]
 [0.00360157]
 [0.00246134]
 [0.0020819 ]
 [0.00221855]
 [0.00360021]
 [0.0033059 ]
 [0.00340318]
 [0.00230613]
 [0.00353658]
 [0.00336196]
 [0.00306764]
 [0.00218437]
 [0.00225152]
 [0.00267865]
 [0.00307053]
 [0.00320347]
 [0.00286234]
 [0.00256801]
 [0.00407442]
 [0.00314877]
 [0.00334164]
 [0.00210665]
 [0.00331722]
 [0.00429094]
 [0.00328143]
 [0.005162  ]
 [0.00330054]
 [0.00261016]
 [0.00457486]
 [0.00259391]
 [0.00232695]
 [0.00337056]
 [0.00376174]
 [0.00468479]
 [0.00449092]
 [0.0042043 ]
 [0.00421794]
 [0.00224024]
 [0.00343156]
 [0.00255793]
 [0.00260165]
 [0.0030143 ]
 [0.0027156 ]
 [0.00275283]
 [0.0024543 ]
 [0.00285951]
 [0.00214399]
 [0.0022919 ]
 [0.00304594]
 [0.00287899]
 [0.00355893]
 [0.00341863]
 [0.00181945]
 [0.00418304]
 [0.00297488]
 [0.00216463]
 [0.00380441]
 [0.00231707]
 [0.00223867]
 [0.00183965]
 [0.0036942 ]
 [0.00270131]
 [0.0036422 ]
 [0.00278167]
 [0.00252328]
 [0.00294933]
 [0.0038947 ]
 [0.00244622]
 [0.00429104]
 [0.00326151]
 [0.00262471]
 [0.00381634]]
 
#### Predict Test output from Created Model

test_predict \ n [[0.00240314]
 [0.0028689 ]
 [0.00285226]
 [0.00214483]
 [0.00281631]
 [0.00432222]
 [0.00355668]
 [0.00237422]
 [0.00378018]
 [0.00401593]
 [0.00237637]
 [0.00419731]
 [0.00382125]
 [0.00253555]
 [0.00364894]
 [0.00243164]
 [0.00312112]
 [0.00269371]
 [0.00320484]
 [0.00240457]
 [0.00295116]
 [0.00227745]
 [0.00386674]
 [0.00256258]
 [0.00301694]
 [0.00507082]
 [0.00249582]
 [0.00288746]
 [0.00244522]
 [0.00246995]
 [0.00293542]
 [0.00374709]
 [0.00321454]
 [0.00500615]
 [0.00371057]
 [0.00384199]
 [0.00269957]
 [0.00345093]
 [0.00390334]
 [0.00356671]
 [0.00259511]
 [0.00261656]
 [0.00271498]
 [0.00218859]
 [0.00478093]
 [0.00385582]
 [0.00316366]
 [0.00222492]
 [0.00441507]
 [0.00502588]]
 
####Test Error percentage  26.004505833456403 %

#### Train Error percentage 26.79232304185279 %

