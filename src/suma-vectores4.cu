#include <stdio.h>

#define N 600
#define tb 500

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  DC[i] = DA[i] + DB[i];
}


int main()
{ 
  cudaFree(0);
  int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;
  int i, dg; int size = N*sizeof(int);

  // reservamos espacio en la memoria global del device
  cudaError_t testerr;
  testerr = cudaMalloc((void**)&DA, size);
  if (testerr!= cudaSuccess) {
	  printf("Error en cudaMalloc DA: %s\n", cudaGetErrorString(testerr));
	  exit(0);
  }	
 
  testerr = cudaMalloc((void**)&DB, size);
  if (testerr!= cudaSuccess) {
	  printf("Error en cudaMalloc DB: %s\n", cudaGetErrorString(testerr));
	  exit(0);
  }		
 
  testerr = cudaMalloc((void**)&DC, size);
  if (testerr!= cudaSuccess) {
	  printf("Error en cudaMalloc DC: %s\n", cudaGetErrorString(testerr));		
	  exit(0);
  }
    
  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
 
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  testerr = cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  if (testerr != cudaSuccess) {
	  printf("Error en cudaMemcpy del host al device: %s\n", cudaGetErrorString(testerr));		
	  exit(0);
  }
 
  testerr = cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  if (testerr != cudaSuccess) {
	  printf("Error en cudaMemcpy del host al device: %s\n", cudaGetErrorString(testerr));		
	  exit(0);
  }      
      
  dg = (N+tb-1)/tb; if (dg>65535) dg=65535;
  // llamamos al kernel
  VecAdd <<dg, tb>>>(DA, DB, DC);	// N o más hilos ejecutan el kernel en paralelo

  testerr = cudaGetLastError();
  if (testerr!= cudaSuccess) {
    printf("Error al ejecutar el kernel: %s\n", cudaGetErrorString(testerr));
	  exit(0);
  }    
  
  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  testerr = cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  if (testerr != cudaSuccess) {
	  printf("Error en cudaMemcpy del device al host: %s\n", cudaGetErrorString(testerr));		
	  exit(0);  
  }
   
  // liberamos la memoria reservada en el device
  cudaFree(DA); cudaFree(DB); cudaFree(DC);  
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i = 0; i < N; i++){
    printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}
  }
  
  return 0;
} 
