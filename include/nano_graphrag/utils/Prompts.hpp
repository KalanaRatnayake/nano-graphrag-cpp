#pragma once

#include <string>

namespace nano_graphrag
{

struct Prompts
{
  static inline const std::string fail_response = "Sorry, I'm not able to provide an answer to that "
                                                  "question.";

  static inline std::string naive_rag_response(const std::string& content_data,
                                               const std::string& response_type)
  {
    return std::string("You're a helpful assistant\n"
                       "Below are the knowledge you know:\n" +
                       content_data +
                       "\n---\n"
                       "If you don't know the answer or if the provided knowledge do not contain sufficient "
                       "information to provide an answer, just say so. Do not make anything up.\n"
                       "Generate a response of the target length and format that responds to the user's "
                       "question, summarizing all information in the input data tables appropriate for the "
                       "response length and format, and incorporating any relevant general knowledge.\n"
                       "If you don't know the answer, just say so. Do not make anything up.\n"
                       "Do not include information where the supporting evidence for it is not provided.\n"
                       "---Target response length and format---\n" +
                       response_type);
  }
};

}  // namespace nano_graphrag
