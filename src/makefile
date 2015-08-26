CC = g++
CFLAGS = -Wall -std=c++11
EXEC_NAME_1 = training.out
EXEC_NAME_2 = testing.out
EXEC_NAME_2 = testing_single_file.out
INCLUDES =
LIBS = `pkg-config --libs opencv`
OBJ_FILES_1 = Tools.o Soft.o ImageData.o Training.o
OBJ_FILES_2 = Tools.o Soft.o Testing.o
OBJ_FILES_3 = Tools.o Soft.o TestingSingleFile.o

all : $(EXEC_NAME_1) $(EXEC_NAME_2) $(EXEC_NAME_3)

clean :
	rm *.o
	
cleanall :
	rm *.out *.o

$(EXEC_NAME_1) : $(OBJ_FILES_1)
	$(CC) -o $(EXEC_NAME_1) $(OBJ_FILES_1) $(LIBS)

$(EXEC_NAME_2) : $(OBJ_FILES_2)
	$(CC) -o $(EXEC_NAME_2) $(OBJ_FILES_2) $(LIBS)

$(EXEC_NAME_3) : $(OBJ_FILES_3)
	$(CC) -o $(EXEC_NAME_3) $(OBJ_FILES_3) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<
