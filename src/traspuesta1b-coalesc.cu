/* Copiar traspuesta de matriz h_a[F][C] en matriz h_b[C][F] aunque el n.º de hebras de 
   los bloques no divida al n.º de componentes de las matrices */
// #include <stdlib.h>
#include <stdio.h>

#define F 25
#define C 43
// matriz original de F filas y C columnas
#define H 8
#define K 8
// bloques de H x K hebras (HxK<=512, cap. cpto. 1.3)


 __global__ void trspta1(int *dev_a, int *dev_b, int filas, int cols){
      
  int ix = blockIdx.x * 8 + threadIdx.x;
  int iy = blockIdx.y * 32 + threadIdx.y;

  __shared__ int s[32][8];

  for(int j = 0; j < 32; j+=8)
      if ((ix < cols) && ((iy+j) < filas))
        s[threadIdx.y + j][threadIdx.x] = dev_a[(iy+j)*cols+ix];
    
  ix = blockIdx.y * 32 + threadIdx.x;
  iy = blockIdx.x * 8 + threadIdx.y;

  for(int j = 0; j < 32; j+=8){
    if (((ix+j) < filas) && (iy < cols)){
      dev_b[iy*filas+(ix+j)] = s[threadIdx.x + j][threadIdx.y];
    }
  }
  
}


int main(int argc, char** argv)
{

  int h_a[F][C], h_b[C][F];
  int *d_a, *d_b;
  int i, j, aux, size = F * C * sizeof(int);
  dim3 hebrasBloque(K, H); // bloques de H x K hebras
 
  int numBlf = (F+32-1)/32;
  int numBlc = (C+8-1)/8;

  printf("nº filas de bloques%d nº columnas de bloques %d", numBlf, numBlc);
 
  dim3 numBloques(numBlc,numBlf);

  // reservar espacio en el device para d_a y d_b
  cudaMalloc((void**) &d_a, size); 
  cudaMalloc((void**) &d_b, size);

  // dar valores a la matriz h_a en la CPU e imprimirlos
  printf("\nMatriz origen\n");
  for (i=0; i<F; i++) {
    for (j=0; j<C; j++) {
      aux = i*C+j;
      h_a[i][j] = aux;
      printf("%d ", aux);
    }
    printf("\n");
  }

  // copiar matriz h_a en d_a
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  
  // llamar al kernel que obtiene en d_b la traspuesta de d_a
  trspta1<<<numBloques, hebrasBloque>>>(d_a, d_b, F, C);

  // copiar matriz d_b en h_b
  cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i=0; i<F; i++)
    for (j=0; j<C; j++) 
      if (h_a[i][j]!= h_b[j][i]) 
		{printf("error en componente %d %d de matriz de entrada \n", i,j); break;}
 
// imprimir matriz resultado
  printf("\nMatriz resultado\n");
  for (i=0; i<C; i++) {
    for (j=0; j<F; j++) {
      printf("%d ", h_b[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  cudaFree(d_a); cudaFree(d_b);
  
  return 0;
}
