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
## Exercice 2 - CPU/GPU memory transfer using CUDA
1) static copy:
![alt text](img/exo2_static.png)
2) dynamic copy:
![alt text](img/exo2_dynamic.png)


&nbsp;  
## Exercice 3 - Writing a GPU-GPU memcpy kernel
On utilise a nouveaux `cudaMemcpy` (with `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`), `cudaMalloc` et `cudaFree` pour allouer et libérer la mémoire sur le GPU, et pour copier les données entre les tableaux sur le GPU. 
Les ids corrects sont `idx = blockIdx.x * blockDim.x + threadIdx.x;`


&nbsp;  
## Exercice 4 - CUDA saxpy kernel
On a 3 cas :
- pour 1 thread par block et une opération par thread, on fait simplement `y[idx] = a * x[idx] + y[idx];` dans le kernel, et on lance `numBlocks = (N + blockSize - 1) / blockSize;` blocks avec `blockSize = 1;`
- pour un certain nombre de threads par block et une opération par thread, on fait la même chose, mais en utilisant `blockSize` threads par block, et en lançant `numBlocks = (N + blockSize - 1) / blockSize;` blocks
- pour un certain nombre de threads par block et K opérations par thread, on fait `y[i] = a * x[i] + y[i];` pour `i` allant de `start` à `end`, où `start = idx * k;` et `end = min(start + k, N);`, et on lance `numBlocks = (N + blockSize * k - 1) / (blockSize * k);` blocks avec `blockSize` threads par block et `k` opérations par thread.




&nbsp;  
## Exercice 5 - CUDA convolution kernel
Même chose que pour l'exercice 4, mais avec la formule de convolution : `y[i] = (x[i - 1] + x[i] + x[i + 1]) / 3.0f;` pour `1 <= i < N-1`, et `y[0] = x[0];` et `y[N-1] = x[N-1];`.  
On voit qu'on peut faire une moyenne glissante malgré la division en blocs, car chaque thread peut accéder à la mémoire globale pour lire les éléments nécessaires à la convolution. Cependant, il faut faire attention aux conditions de bord, et s'assurer que les threads qui traitent les éléments aux extrémités du tableau ne tentent pas d'accéder à des indices hors limites.  
