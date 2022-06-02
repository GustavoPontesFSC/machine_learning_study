"""
SVM - Support Vector Machine
Como nos artigos estavam indicando um melhor desempenho acredito que esta seja a melhor opção para estudar e me dedicar.
Atualmente estou conseguindo manipular objetos com características definidas
"""

"""
Na primeira parta faço a importação do numpy para a criação dos array's e importo alguns modulos da biblioteca sklearn
a parte do sk learn não estou completamente entendido de como funciona mas por hora acho que o importante e saber o porque do funcionamento.
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


"""
Importado os móduilos criaremos os arrays que serão nossa amostra, no exemplo abaixo estou descrevendo um modelo simples no qual diz uma população ficticia 
de animais no qual as femeas adultas teriam o tamanho médio de 130 cm e os machos de 150, por sua vez as femeas teriam uma garra de tamnho médio de 
15 cm enquanto os machos de 4 cm e também os machos costumam dormir em média 8h enquanto as femeas 6h
"""
X = np.array([[133, 15, 6.4],[162,3, 8.2],[129,13.2,5.9],[151,4,7.8],[132,16,7.5],[143,5,8]])
y = np.array([1, 0, 1, 0, 1, 0]) #Defini 0 = macho 1 = femea
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X, y)
"""
Feito o modelo de analise agora é hora de fazer a predição de algum dado vamos ver para dois casos e um com uma forma um pouco diferente
"""

predict1 = int(clf.predict([[144, 3,7.3]]))
if predict1 == 0:
    print('Macho')
else:
    print('Femea')
print('Vemos aqui um com as caracteristicas padrão de um macho e o print acabou analisando corretamente')

print('---------')

predict2 = int(clf.predict([[135, 4,7.7]]))

if predict2 == 0:
    print('Macho')
else:
    print('Femea')
print('Aqui por outro lado tem tamanho de femea, porém a garra e o preiodo do sono é de macho então a maquina reconheceu como macho')
print('---------')

predict3 = int(clf.predict([[135, 12.9 ,5.7]]))

if predict3 == 0:
    print('Macho')
else:
    print('Femea')
print('Enquanto aqui a maquina ja conseguiu analisar como femea')