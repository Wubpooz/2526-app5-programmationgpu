# TP1 - Introduction to CUDA Programming
**Mathieu WAHARTE - 17/03/2026**


## Exercice 1 - Hello World
Pour un blocSize de 1 et 64 blocks on a :  
![alt text](img/exo1_b1.png)

Pour un blockSize de 8 et 64 blocks on a :
![alt text](img/exo1_b8.png)

Pour un blockSize de 16 et 64 blocks on a :
![alt text](img/exo1_b16.png)

Pour un blockSize de 64 et 64 blocks on a :
![alt text](img/exo1_b64.png)

On peut voir que les threads sont regroupés par block, et que les threads d'un même block ont des threadIdx allant de 0 à blockSize-1. Les blocks sont numérotés de 0 à numBlocks-1.  


&nbsp;  
## Exercice 2 - Addition de vecteurs
1) static copy:
![alt text](img/exo2_static.png)
2) dynamic copy:
![alt text](img/exo2_dynamic.png)