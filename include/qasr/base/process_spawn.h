#pragma once

#include <string>
#include <vector>

namespace qasr {

/// Spawn the executable named by `args[0]` (searched via the system
/// PATH) and wait for it to terminate.  Each element of `args` is
/// passed as a separate argument to the new process; the call does
/// NOT go through a shell, so shell metacharacters in any argument
/// cannot be interpreted as commands.
///
/// Returns the process exit code on success (0 means the program
/// exited normally with status 0).  Returns -1 on spawn failure
/// (e.g. the executable could not be found, or insufficient
/// permissions).  If the process was killed by a signal, returns
/// 128 + signal number (POSIX) or -1 (Windows).
int SpawnAndWait(const std::vector<std::string> & args);

}  // namespace qasr
