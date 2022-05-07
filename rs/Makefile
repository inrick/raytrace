.PHONY: all run gui
#RUSTFLAGS := -A dead_code -A unused_variables
RUSTFLAGS :=
all:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo build
run:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --release
gui:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --release --features=gui
