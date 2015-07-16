CC = g++
CFLAGS = -Wall -std=c++11
EXEC_NAME = vocabulary
INCLUDES =
LIBS = `pkg-config --libs opencv`
OBJ_FILES = tools.o vocabulary.o

all : $(EXEC_NAME)

clean :
	rm *.o
	
cleanall :
	rm *.exe *.o

$(EXEC_NAME) : $(OBJ_FILES)
	$(CC) -o $(EXEC_NAME) $(OBJ_FILES) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $<
