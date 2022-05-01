.PHONY: all math32 math64 tinygo
all: math32

math32:
	go build && time ./ray > test.ppm

math64:
	go build -tags math64 && time ./ray > test.ppm

# tinygo produces a significantly faster executable
tinygo:
	tinygo build -o ray . && time ./ray > test.ppm
