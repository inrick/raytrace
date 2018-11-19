BIN := ray

.PHONY: all clean
all:
	clang -g3 -O0 -std=c99 -Wall -fno-strict-aliasing -o $(BIN) main.c

clean:
	-rm -f $(BIN)
