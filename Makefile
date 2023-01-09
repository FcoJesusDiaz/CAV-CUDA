DIREXE := exec/
DIRSRC := src/

CC := nvcc

all : dirs compile

dirs:
	mkdir -p $(DIREXE) 

compile:
	$(CC) $(DIRSRC)suma-vectores1.cu -o $(DIREXE)suma-vectores1
	$(CC) $(DIRSRC)suma-vectores1b.cu -o $(DIREXE)suma-vectores1b
	$(CC) $(DIRSRC)suma-vectores2.cu -o $(DIREXE)suma-vectores2
	$(CC) $(DIRSRC)suma-vectores3.cu -o $(DIREXE)suma-vectores3
	$(CC) $(DIRSRC)suma-vectores4.cu -o $(DIREXE)suma-vectores4
	$(CC) $(DIRSRC)traspuesta1.cu -o $(DIREXE)traspuesta1
	$(CC) $(DIRSRC)traspuesta1-coalesc.cu -o $(DIREXE)traspuesta1-coalesc
	$(CC) $(DIRSRC)traspuesta1b-coalesc.cu -o $(DIREXE)traspuesta1-coalesc
	$(CC) $(DIRSRC)traspuesta2.cu -o $(DIREXE)traspuesta2
	$(CC) $(DIRSRC)traspuesta3.cu -o $(DIREXE)traspuesta3

clean : 
	rm -rf *~ core $(DIREXE) $(DIRSRC)*~ 
