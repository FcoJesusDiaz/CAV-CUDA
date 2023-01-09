DIREXE := exec/
DIRSRC := src/

CC := nvcc

all : dirs compile

dirs:
	mkdir -p $(DIREXE) 

compile:
	$(CC) $(DIRSRC)suma-vectores.cu -o $(DIREXE)suma-vectores
	$(CC) $(DIRSRC)suma-vectores1b.cu -o $(DIREXE)suma-vectores1b
	$(CC) $(DIRSRC)suma-vectores2.cu -o $(DIREXE)suma-vectores2
	$(CC) $(DIRSRC)suma-vectores3.cu -o $(DIREXE)suma-vectores3
	$(CC) $(DIRSRC)suma-vectores4.cu -o $(DIREXE)suma-vectores4
	$(CC) $(DIRSRC)trapuesta1.cu -o $(DIREXE)trapuesta1
	$(CC) $(DIRSRC)trapuesta2.cu -o $(DIREXE)trapuesta2
	$(CC) $(DIRSRC)trapuesta3.cu -o $(DIREXE)trapuesta3

clean : 
	rm -rf *~ core $(DIREXE) $(DIRSRC)*~ 
