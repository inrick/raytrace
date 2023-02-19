The Go implementation can be built by running `make` or `go build`. It can then
be run through the commandline:

```
./ray -n 100 -o out.png
```

(Unlike the C and Rust implementations, the number of threads cannot be
specified. This is because the Go implementation uses a goroutine per line in
the output image and lets the Go runtime handle parallelization.)
