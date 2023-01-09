#include <stdio.h>

#define N 600

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	int i = blockIdx.x;
    DC[i] = DA[i] + DB[i];
}

int main()
{ int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;
  int i; int size = N*sizeof(int);
  cudaError_t error;
  
  // reservamos espacio en la memoria global del device
  error = cudaMalloc((void**)&DA, size);
  if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
  error = cudaMalloc((void**)&DB, size);
  if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
  error = cudaMalloc((void**)&DC, size);
  if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
  
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  error = cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
  error = cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
  
  // llamamos al kernel (1 bloque de N hilos)
  VecAdd <<<N, 1>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo
  
  cudaError_t error_synchro = cudaGetLastError();
  cudaError_t error_asynchro = cudaDeviceSynchronize();
  if (error_synchro != cudaSuccess) printf("Sync kernel error: %s\n", cudaGetErrorString(error_synchro));
  if (error_asynchro != cudaSuccess) printf("Async kernel error: %s\n", cudaGetErrorString(error_asynchro));
  
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  error = cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) printf("%s\n", cudaGetErrorString(error));
  
  // liberamos la memoria reservada en el device
  cudaFree(DA); cudaFree(DB); cudaFree(DC);  
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  // esta comprobación debe quitarse una vez que el programa es correcto (p. ej., para medir el tiempo de ejecución)
  for (i = 0; i < N; i++){
    //printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{
        printf("error en componente %d\n", i);
        break;}
  }
    
  return 0;
} 
