.PHONY: all run
#RUSTFLAGS := -A dead_code -A unused_variables
RUSTFLAGS :=
all:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo --quiet build
run:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo --quiet run --release
