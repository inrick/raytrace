.PHONY: all run gui profile
#RUSTFLAGS := -A dead_code -A unused_variables
RUSTFLAGS := -C target-cpu=native
all:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo build --frozen
run:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --frozen --release -- -o out.png -n 100
gui:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --frozen --release --features=gui
profile:
	@RUSTFLAGS="$(RUSTFLAGS)" cargo run --frozen --profile=release-with-debug --features=pprof -- -o out.png -n 100 --cpuprof tmp/cpuprof.pb
