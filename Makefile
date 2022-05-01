.PHONY: all tinygo
all:
	go build && time ./ray > test.ppm

# tinygo produces a significantly faster executable
tinygo:
	tinygo build -o ray . && time ./ray > test.ppm
