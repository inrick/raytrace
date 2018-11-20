BIN := ray

# make bin phony to always rebuild
.PHONY: all clean $(BIN)
all: $(BIN)
	./$(BIN) > test.ppm

$(BIN):
	clang -g3 -O3 -std=gnu99 -Wall -Wextra -fno-strict-aliasing -o $(BIN) main.c -lm

clean:
	-rm -f $(BIN)
