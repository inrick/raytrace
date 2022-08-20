CFLAGS := \
  -march=native \
  -std=gnu11 \
  -Wall \
  -Wextra \
  -Wswitch-enum \
  -fno-strict-aliasing \
  -fwrapv
CFLAGS_DEBUG   := -O0 -g3 $(CFLAGS)
CFLAGS_RELEASE := -O2 $(CFLAGS)
LDFLAGS := -lm -lpthread
BIN := ray

# make bin phony to always rebuild
.PHONY: all clean $(BIN) debug asm run
all: $(BIN)

$(BIN):
	clang $(CFLAGS_RELEASE) -o $(BIN) main.c $(LDFLAGS)

debug:
	clang $(CFLAGS_DEBUG) -o $(BIN) main.c $(LDFLAGS)

asm:
	clang -S -masm=intel $(CFLAGS_RELEASE) -o main.s main.c

run: $(BIN)
	time ./$(BIN) -oout.png -n100 -t8

clean:
	-rm -f $(BIN) main.s
