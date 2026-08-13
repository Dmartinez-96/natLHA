// Entry point for the interactive natLHA executable.
//
// Kept in a file of its own containing nothing else. An object file that defines main()
// cannot be linked into another executable, so keeping main() out of the sources holding
// the physics lets those sources be built once as a library and reused by other programs.
//
// Keep this file trivial: anything beyond starting the terminal UI belongs in the shared
// sources, where other callers can reach it.

#include "terminal_UI.hpp"

int main() {
    terminalUI();
    return 0;
}
