CC = g++
CFLAGS = -Wall -std=c++11
EXEC_NAME = training.out
INCLUDES =
LIBS = `pkg-config --libs opencv`
OBJ_FILES = Tools.o Soft.o ImageData.o Training.o

all : $(EXEC_NAME)

clean :
	rm *.o
	
cleanall :
	rm *.out *.o

$(EXEC_NAME) : $(OBJ_FILES)
	$(CC) -o $(EXEC_NAME) $(OBJ_FILES) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<
