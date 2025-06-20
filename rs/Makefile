.PHONY: all run gui
#RUSTFLAGS := -A dead_code -A unused_variables
RUSTFLAGS :=
all:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo build --frozen
run:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --frozen --release -- -o out.png -n 100
gui:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --frozen --release --features=gui
