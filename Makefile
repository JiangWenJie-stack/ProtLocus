CC=			gcc
CXX=		g++
# -fPIE: some toolchains' ld defaults to linking a PIE executable, which requires
# every object to be compiled PIE-safe or it fails with "relocation R_X86_64_32S
# against `.rodata' can not be used when making a PIE object".
CFLAGS=		-std=c99 -g -Wall -O3 -fPIE
CXXFLAGS=	$(CFLAGS)
CPPFLAGS=
INCLUDES=
LDFLAGS=
OBJS=		kalloc.o kthread.o nasw-sse.o nasw-tab.o sys.o misc.o options.o hmm.o hmm-gpu.o ntseq.o sketch.o \
			nasw-opencl.o nasw-ext-gpu.o index.o bseq.o chain.o hit.o format.o map.o align.o prescreen.o
PROG=		protlocus
LIB=		libprotlocus.a
LIBS=		-lpthread -lz -lm

ifneq ($(asan),)
	CFLAGS+=-fsanitize=address
	LIBS+=-fsanitize=address -ldl
endif

# static=1: fully static CPU executable (needs static libz/libc); not compatible with gpu=1/cuda=1
ifneq ($(static),)
	LDFLAGS+=-static
endif

ifneq ($(no_avx2),)
	override CPPFLAGS+=-DNS_DISABLE_AVX2_DISPATCH
endif

ifneq ($(gpu),)
	override CPPFLAGS+=-DNS_OPENCL
	LIBS+=-lOpenCL
endif

ifneq ($(cuda),)
	NVCC?=nvcc
	# auto-detect from the first visible GPU; override with e.g. CUDA_ARCH=sm_89
	DETECTED_CUDA_ARCH:=$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
	CUDA_ARCH?=$(if $(DETECTED_CUDA_ARCH),sm_$(DETECTED_CUDA_ARCH),sm_80)
	CUDA_GENCODE?=-arch=$(CUDA_ARCH)
CUDA_HOME?=$(patsubst %/bin/nvcc,%,$(shell command -v $(NVCC)))
	override CPPFLAGS+=-DNS_CUDA -I$(CUDA_HOME)/include
	OBJS:=$(filter-out nasw-opencl.o,$(OBJS)) nasw-cuda.o
	LIBS+=-L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lcudart -lstdc++
endif

.SUFFIXES:.c .cpp .cu .o
.PHONY:all clean depend check

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

.cpp.o:
		$(CXX) -c $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

.cu.o:
		$(NVCC) -c $(CUDA_GENCODE) $(NVCCFLAGS) -O3 -Xcompiler -Wall,-fPIE $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(PROG) protlocus-recheck

$(LIB):$(OBJS)
		$(AR) -csru $@ $(OBJS)

protlocus:$(OBJS) main.o $(LIB)
		$(CC) $(CFLAGS) $(LDFLAGS) main.o -o $@ -L. -lprotlocus $(LIBS)

# native candidate + HMM recheck tool (5.0.0); CPU only, no HMMER dependency
protlocus-recheck:recheck.o kthread.o
		$(CC) $(CFLAGS) $(LDFLAGS) recheck.o kthread.o -o $@ -lpthread -lz -lm

protlocus-lite:example.o $(LIB)
		$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ -L. -lprotlocus $(LIBS)

test/test_dp32:test/test_dp32.c $(LIB)
		$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(LDFLAGS) -I. $< -o $@ -L. -lprotlocus $(LIBS)

test/test_accuracy:test/test_accuracy.c $(LIB)
		$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(LDFLAGS) -I. $< -o $@ -L. -lprotlocus $(LIBS)

test/test_gpu:test/test_gpu.c $(LIB)
		$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $(LDFLAGS) -I. $< -o $@ -L. -lprotlocus $(LIBS)

check:$(PROG) test/test_dp32 test/test_accuracy test/test_gpu
		./test/test_dp32
		./test/test_accuracy
		./test/test_gpu
		python3 -m unittest discover -s test -p 'test_*.py'

clean:
		rm -fr *.o a.out $(PROG) protlocus-lite protlocus-recheck *~ *.a *.dSYM test/test_dp32 test/test_accuracy test/test_gpu

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c *.cpp)

# DO NOT DELETE

align.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h
bseq.o: kvec-km.h kalloc.h mppriv.h protlocus.h nasw.h kseq.h
chain.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h
example.o: protlocus.h nasw.h kalloc.h kseq.h
format.o: kseq.h mppriv.h protlocus.h nasw.h kalloc.h
hit.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h
hmm.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h
index.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h kvec-km.h kthread.h
kalloc.o: kalloc.h
kthread.o: kthread.h
main.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h ketopt.h
map.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h kthread.h kvec-km.h
misc.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h ksort.h
nasw-sse.o: nasw.h kalloc.h
nasw-opencl.o: nasw.h kalloc.h
nasw-cuda.o: nasw.h kalloc.h
nasw-tab.o: nasw.h kalloc.h
hmm-gpu.o: protlocus.h
ntseq.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h khashl-km.h
options.o: protlocus.h nasw.h kalloc.h
prescreen.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h kthread.h kvec-km.h
sketch.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h kvec-km.h
sys.o: mppriv.h protlocus.h nasw.h kalloc.h kseq.h
