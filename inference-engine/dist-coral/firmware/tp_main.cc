// Firmware entry point for one rank of a tensor-parallel GPT-Neo: serve the
// host (dist_coral/tp_host_if.hpp) until it says stop, then halt. crt0 runs
// main after reset and executes mpause when it returns.

#include "dist_coral/tp_host_if.hpp"
#include "tp_rank.hpp"

// The linker script puts .mailbox first in DTCM, at kMailboxAddress.
__attribute__((section(".mailbox"), used)) dist_coral::TpMailbox mailbox;

int main() {
    firmware::serve(mailbox);
    return 0;
}
