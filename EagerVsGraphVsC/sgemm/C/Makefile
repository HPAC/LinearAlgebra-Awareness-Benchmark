SRC     := src
OBJ     := obj
BIN     := bin

CC      := gcc
LINKER  := $(CC)

#MKL_INC := -m64 -I${MKLROOT}/include

DMKL_ILP64 := -m64 -I${MKLROOT}/include
INC     := -Isrc/timer $(MKL_INC)
CFLAGS  := -O3 -march=native -Wall $(INC)

#MKL_LD  := -L${MKLROOT}/lib/ -Wall -Wl, -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core\
#	         -lpthread -lm -ldl

MKL_LD  :=  -L${MKLROOT}/lib -L /opt/intel/oneapi/compiler/latest/mac/compiler/lib/ -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

#Use this linker in cluster
#MKL_LD  :=  -L${MKLROOT}/lib -L ${MKLROOT}/../lib/intel64 -Wl,-rpath,${MKLROOT}/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

LDFLAGS := $(MKL_LD)

DEFS    := -DLAMP_REPS=$(LAMP_REPS) -DLAMP_L3_CACHE_SIZE=$(LAMP_L3_CACHE_SIZE)

SOURCES  := $(wildcard $(SRC)/*.c)
BINARIES := $(patsubst $(SRC)/%.c, $(BIN)/%.x, $(SOURCES))

default: all
all: $(BINARIES)

$(OBJ)/%.o: $(SRC)/timer/%.c
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

$(OBJ)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) $(DEFS) -c $< -o $@

$(BIN)/%.x: $(OBJ)/%.o $(OBJ)/timer.o
	$(LINKER) $< $(OBJ)/timer.o $(LDFLAGS) -o $@

clean:
	rm -f $(BIN)/* $(OBJ)/*
