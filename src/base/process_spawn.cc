#include "qasr/base/process_spawn.h"

#ifdef _WIN32
#  include <errno.h>
#  include <process.h>
#else
#  include <spawn.h>
#  include <sys/wait.h>
#  include <unistd.h>
extern char ** environ;
#endif

namespace qasr {

int SpawnAndWait(const std::vector<std::string> & args) {
    if (args.empty()) {
        return -1;
    }
#ifdef _WIN32
    std::vector<const char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);
    intptr_t rc = _spawnvp(_P_WAIT, args[0].c_str(),
                           const_cast<char * const *>(argv.data()));
    if (rc == -1) {
        return -1;
    }
    return static_cast<int>(rc);
#else
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const auto & a : args) {
        argv.push_back(const_cast<char *>(a.c_str()));
    }
    argv.push_back(nullptr);
    pid_t pid = -1;
    int spawn_rc = posix_spawnp(&pid, args[0].c_str(), nullptr, nullptr,
                                argv.data(), environ);
    if (spawn_rc != 0) {
        return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    if (!WIFEXITED(status)) {
        return 128 + WTERMSIG(status);
    }
    return WEXITSTATUS(status);
#endif
}

}  // namespace qasr
