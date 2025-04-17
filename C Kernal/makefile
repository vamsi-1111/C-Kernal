all: libkernel.so

libkernel.so: kernel.c
	gcc -O3 -shared -fPIC -o libkernel.so kernel.c -lm

clean:
	rm -f libkernel.so

