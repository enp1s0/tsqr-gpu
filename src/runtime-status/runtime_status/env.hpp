#ifndef __RUNTIME_STATUS_ENV_HPP__
#define __RUNTIME_STATUS_ENV_HPP__
#include <string>
#include <stdlib.h>

namespace mtk {
namespace runtime_status {
namespace env {
void print_runtime_env(const std::string env_key) {
	const auto env = getenv(env_key.c_str());
	std::printf("%13s : %s\n", env_key.c_str(), env);
}
} // namespace env
} // namespace runtime_status
} // namespace mtk
#endif
