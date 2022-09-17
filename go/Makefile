.PHONY: all math32 math64 tinygo
all: math32

math32:
	go build && time ./ray -o out.png

math64:
	go build -tags math64 && time ./ray -o out.png

# tinygo produces a significantly faster executable for single threaded runs
# but forces GOMAXPROCS=1
tinygo:
	tinygo build -o ray . && time ./ray -o out.png
