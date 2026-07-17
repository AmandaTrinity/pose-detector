# normalmente, não usamos as coordenadas exatas(dist, posicao,altura...) e sim o ângulo das articulações
import numpy as np
def angulo_corpo(c1,c2,c3):

    # transformamos as coordenadas em vetores
    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)

    # c2 é articulação
    v1 = c1 - c2
    v2 = c3 - c2
    
    # formula do produto escalar
    # cos(ang) = (v1.v2)/(||v1||*||v2||)
    # ang = arccos()

    # dot faz a multiplicação de matrizes
    # norm faz a norma (tamanho) do vetor
    # linalg é uma biblioteca de álgebra linear
    angulo = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    # convertendo para graus
    return np.degrees(angulo)
    