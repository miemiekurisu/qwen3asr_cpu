#include "qasr/service/server.h"

#include <iostream>

int main(int argc, char ** argv) {
    qasr::ServerConfig config;
    bool show_help = false;
    const qasr::Status status = qasr::ParseServerArguments(argc, argv, &config, &show_help);
    if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        std::cerr << qasr::BuildServerUsage(argc > 0 ? argv[0] : "qasr_server");
        return 1;
    }

    if (show_help) {
        std::cout << qasr::BuildServerUsage(argv[0]);
        return 0;
    }

    return qasr::RunServer(config);
}
