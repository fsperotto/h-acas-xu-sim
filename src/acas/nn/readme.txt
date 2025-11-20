ATTENTION : l'ACAS-XU et les réseaux qui en sont issus sont la propriété de l'EUROCAE
NE PAS DIFFUSER !

Des réseaux relatifs à l'ACAS-XU. 
Le format original est NNET

nnet2onnx a été utilisé pour les traduire en ONNX 
https://github.com/sisl/NNet/blob/master/converters/nnet2onnx.py 

il y a un paramètre qui permet d'intégrer ou pas le pré processing et le post processing dans le calcul 
des paramètres des couches d'entrée et de sortie du réseau. 
normalizeNetwork: (bool) 
If true, adapt the network weights and biases so that networks and inputs do not need to be normalized. 
Default is False.
Bien sûr cette normalisation ne prend pas en compte les saturation de valeurs hors domaine. 
On peut constater que pour le réseau sans besoin de normalization (True) 
la valeur des poids de la matrice W0 associés à la première entrée sont très petites 
car elle peuvent être amenées à être multipliées par des valeurs de distance en mètres qui peuvent être très grandes. 
Ce n'est pas le cas pour le réseau avec besoin de normalization (False)

Netron a été utilisé pour dessiner les graphes 
https://netron.app/ 


exemple d'inférence sur la version originale du réseau (.nnet) :

>>> nnet = NNet('ACASXU_run2a_1_1_batch_2000.nnet')
>>> print("Num Inputs: %d"%nnet.num_inputs())
Num Inputs: 5
>>> print("Num Outputs: %d"%nnet.num_outputs())
Num Outputs: 5
>>> print("One evaluation:")
One evaluation:
>>> print(nnet.evaluate_network([15299.0,0.0,-3.1,600.0,500.0]))
[14.87572516 12.19844507 27.18484146  5.43263927 27.2421045 ] 

En complément, le pré-processing avant l'inférence est :

            line = f.readline() # ligne 4 [0.0,-3.141593,-3.141593,100.0,0.0]
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline() # ligne 5 [60760.0,3.141593,3.141593,1200.0,1200.0]
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline() # ligne 6 [1.9791091e+04,0.0,0.0,650.0,600.0,7.5188840201005975]
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]

            line = f.readline() # ligne 7 [60261.0,6.28318530718,6.28318530718,1100.0,1200.0,373.94992]
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]
...

            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges

...

        # Prepare the inputs to the neural network  - inputs est les entrées non normalisées, inputNorm les entrées normalisées
        inputsNorm = np.zeros(inputSize)
        for i in range(inputSize):
            if inputs[i]<self.mins[i]:
                inputsNorm[i] = (self.mins[i]-self.means[i])/self.ranges[i]
            elif inputs[i]>self.maxes[i]:
                inputsNorm[i] = (self.maxes[i]-self.means[i])/self.ranges[i]
            else:
                inputsNorm[i] = (inputs[i]-self.means[i])/self.ranges[i] 
				
				
Voici l'impression des étapes intermédiaires :

print(nnet.evaluate_network([15299.0,0.0,-3.1,600.0,500.0])) <- entrées brutes
inputsNorm
[-0.07454392  0.         -0.49338032 -0.04545455 -0.08333333] <- entrées du réseau
outputsNorm
[ 0.01967333  0.01251387  0.05258982 -0.00557894  0.05274295] <- sorties du réseau
[14.87572516 12.19844507 27.18484146  5.43263927 27.2421045 ] <- sorties de-normalisées 