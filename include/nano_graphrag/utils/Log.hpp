#pragma once

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace nano_graphrag
{

inline bool debug_enabled()
{
  const char* v = std::getenv("NANO_GRAPHRAG_DEBUG");
  if (!v)
    return false;
  std::string s(v);
  for (auto& c : s)
    c = static_cast<char>(::tolower(c));
  return (s == "1" || s == "true" || s == "yes" || s == "on");
}

template <typename... Args>
inline void debug_log(Args&&... args)
{
  if (!debug_enabled())
    return;
  (std::cerr << ... << args) << std::endl;
}

}  // namespace nano_graphrag
