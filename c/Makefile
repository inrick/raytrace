#CFLAGS := \
#  -march=native \
#  -g3 -O0 \
#  -std=gnu99 \
#  -Wall \
#  -Wextra \
#  -Wswitch-enum \
#  -fno-strict-aliasing
CFLAGS := \
  -march=native \
  -O3 \
  -std=gnu99 \
  -Wall \
  -Wextra \
  -Wswitch-enum \
  -fno-strict-aliasing
BIN := ray

# make bin phony to always rebuild
.PHONY: all clean $(BIN) asm
all: $(BIN)
	time ./$(BIN) -o out.png

$(BIN):
	clang $(CFLAGS) -o $(BIN) main.c -lm

asm:
	clang -S $(CFLAGS) -o main.s main.c

clean:
	-rm -f $(BIN) main.s
