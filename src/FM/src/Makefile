CC=g++

SOURCES=$(TARGET).cpp 

OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=$(TARGET)

#CFLAGS := -Wall -g -O3 -Wfatal-errors -mssse3 -msse2 -march=native
CFLAGS := -ggdb -g `pkg-config opencv4 --cflags`
LDFLAGS :=  grief/grief.cpp -Llib  -g `pkg-config opencv4 --libs` -fopenmp


#CFLAGS += `pkg-config opencv --cflags`

all: match_all

match_all: match_all.cpp grief/grief.cpp 
	$(CC) match_all.cpp $(CFLAGS) $(LDFLAGS) -o $@ 

evolve_grief: evolve_grief.cpp grief/grief.cpp 
	$(CC) evolve_grief.cpp $(CFLAGS) $(LDFLAGS) -o $@
 
annotate: annotate.cpp grief/grief.cpp 
	$(CC) annotate.cpp $(CFLAGS) $(LDFLAGS) -o $@ 

clean:
	rm -f $(OBJECTS) $(EXECUTABLE) 
	rm -f match_all evolve_grief annotate 

cleanall:
	rm -f *.o 
