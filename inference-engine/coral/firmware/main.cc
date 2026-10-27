// Firmware entry point: serve the host's request in the mailbox, then halt.
// crt0 runs main after reset and executes mpause when it returns.

#include "gpt_neo.hpp"
#include "reef_kernels/host_if.hpp"

// The linker script puts .mailbox first in DTCM, at kMailboxAddress.
__attribute__((section(".mailbox"), used)) reef_kernels::Mailbox mailbox;

int main() {
    firmware::serve(mailbox);
    return 0;
}
