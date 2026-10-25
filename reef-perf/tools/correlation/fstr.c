/* fstr: a minimal reader for the FST waveforms of the M3 Verilator simulator.

     fstr list <file.fst>              one line per variable: handle width name
     fstr dump <file.fst> <h1,h2,...>  one line per value change: time handle value

   It uses the FST library bundled with Verilator, the one that wrote the file.
   GTKWave's fst2vcd from Debian rejects these files ("TALOS-2023-1791 security
   check"). Build it in the CoralNPU container:

     G=$(dirname $(find ~/.cache/bazel -path \
           '*external/verilator/include/gtkwave/fstapi.c' | head -1))
     gcc -O2 -I$G -DFST_CONFIG_INCLUDE='"fst_config.h"' fstr.c \
         $G/fstapi.c $G/fastlz.c $G/lz4.c -lz -lpthread -o fstr
*/

#include "fstapi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void printChange(void *User, uint64_t Time, fstHandle Handle,
                        const unsigned char *Value) {
  (void)User;
  printf("%llu %u %s\n", (unsigned long long)Time, Handle, Value);
}

int main(int Argc, char **Argv) {
  if (Argc < 3 || (strcmp(Argv[1], "list") && Argc < 4)) {
    fprintf(stderr, "usage: fstr list <file.fst> | dump <file.fst> <h1,h2,...>\n");
    return 2;
  }
  void *Ctx = fstReaderOpen(Argv[2]);
  if (!Ctx) {
    fprintf(stderr, "fstr: cannot open %s\n", Argv[2]);
    return 1;
  }
  if (!strcmp(Argv[1], "list")) {
    char Path[4096] = "";
    struct fstHier *H;
    while ((H = fstReaderIterateHier(Ctx))) {
      if (H->htyp == FST_HT_SCOPE) {
        if (Path[0])
          strcat(Path, ".");
        strcat(Path, H->u.scope.name);
      } else if (H->htyp == FST_HT_UPSCOPE) {
        char *Dot = strrchr(Path, '.');
        if (Dot)
          *Dot = 0;
        else
          Path[0] = 0;
      } else if (H->htyp == FST_HT_VAR) {
        printf("%u %u %s.%s\n", H->u.var.handle, H->u.var.length, Path,
               H->u.var.name);
      }
    }
  } else {
    fstReaderClrFacProcessMaskAll(Ctx);
    for (char *Tok = strtok(Argv[3], ","); Tok; Tok = strtok(NULL, ","))
      fstReaderSetFacProcessMask(Ctx, (fstHandle)atoi(Tok));
    fstReaderIterBlocks(Ctx, printChange, NULL, NULL);
  }
  fstReaderClose(Ctx);
  return 0;
}
