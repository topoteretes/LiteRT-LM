// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Knowledge graph extraction demo using LiteRT-LM C API with constrained
// decoding.
//
// Given an input text document, prompts the LLM to extract entities (nodes)
// and relationships (edges) and output them as a JSON object conforming to a
// hardcoded schema.
//
// Usage:
//   simple_chat --model_path=/path/to/model.litertlm \
//               --input_text="Alice works at Google. Bob works at Meta." \
//               --backend=cpu

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "c/engine.h"
#include "nlohmann/json.hpp"  // from @nlohmann_json

ABSL_FLAG(std::string, model_path, "", "Path to the .litertlm model file.");
ABSL_FLAG(std::string, backend, "cpu", "Backend to use (cpu, gpu, npu).");
ABSL_FLAG(bool, add_constraint, true, "Enable constrained decoding.");
ABSL_FLAG(std::string, input_text, "", "Input text to extract knowledge from.");
ABSL_FLAG(std::string, input_text_file, "",
          "Path to a file containing the input text.");
ABSL_FLAG(std::string, litert_dispatch_lib_dir, "",
          "Directory of the LiteRT dispatch library (e.g. for NPU/QNN).");

namespace {

using ::nlohmann::json;

// System prompt instructing the LLM to extract knowledge graph from text.
constexpr char kKnowledgeGraphSystemPrompt[] =
    "You are a knowledge graph extraction expert. Your task is to analyze the "
    "provided text and extract a structured knowledge graph from it.\n\n"
    "Extract all named entities as nodes with the following properties:\n"
    "- id: a short snake_case identifier (e.g. 'alice', 'google_inc')\n"
    "- name: the entity's full display name\n"
    "- type: the entity type (e.g. 'Person', 'Organization', 'Location', "
    "'Concept')\n"
    "- description: a brief description of the entity based on the text\n\n"
    "Extract all relationships as edges with the following properties:\n"
    "- source_node_id: id of the source node\n"
    "- target_node_id: id of the target node\n"
    "- relationship_name: a concise label for the relationship "
    "(e.g. 'WORKS_AT', 'LOCATED_IN', 'FOUNDED_BY')\n\n"
    "Output ONLY valid JSON conforming to the provided schema. "
    "Do not include any explanation or markdown.\n\n"
    "JSON Schema:\n";

// JSON Schema (draft-07) for the KnowledgeGraph output type.
constexpr char kKnowledgeGraphSchema[] = R"json({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KnowledgeGraph",
  "type": "object",
  "required": ["nodes", "edges"],
  "additionalProperties": false,
  "properties": {
    "nodes": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Node",
        "required": ["id", "name", "type", "description"],
        "additionalProperties": false,
        "properties": {
          "id": {"type": "string"},
          "name": {"type": "string"},
          "type": {"type": "string"},
          "description": {"type": "string"}
        }
      }
    },
    "edges": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Edge",
        "required": ["source_node_id", "target_node_id", "relationship_name"],
        "additionalProperties": false,
        "properties": {
          "source_node_id": {"type": "string"},
          "target_node_id": {"type": "string"},
          "relationship_name": {"type": "string"}
        }
      }
    }
  }
})json";

// Reads all text from the given file path.
std::string ReadTextFile(const std::string& path, bool* success) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << path << std::endl;
    *success = false;
    return "";
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  *success = true;
  return buffer.str();
}

// Compacts a JSON schema string (removes whitespace) for embedding in prompts.
std::string CompactJsonSchemaForPrompt(const std::string& schema_str) {
  try {
    return json::parse(schema_str).dump();
  } catch (...) {
    return schema_str;
  }
}

// Attempts to extract the first complete {...} JSON object from a string.
// Falls back to the full string if no balanced braces are found.
std::string ExtractFirstJsonObject(const std::string& text) {
  int depth = 0;
  std::size_t start = std::string::npos;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '{') {
      if (depth == 0) start = i;
      ++depth;
    } else if (text[i] == '}') {
      --depth;
      if (depth == 0 && start != std::string::npos) {
        return text.substr(start, i - start + 1);
      }
    }
  }
  return text;
}

int Run(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    std::cerr << "--model_path is required." << std::endl;
    return 1;
  }
  const std::string input_text = absl::GetFlag(FLAGS_input_text);
  const std::string input_text_file = absl::GetFlag(FLAGS_input_text_file);
  if (input_text.empty() && input_text_file.empty()) {
    std::cerr << "One of --input_text or --input_text_file is required."
              << std::endl;
    return 1;
  }
  if (!input_text.empty() && !input_text_file.empty()) {
    std::cerr
        << "Only one of --input_text or --input_text_file may be specified."
        << std::endl;
    return 1;
  }

  std::string source_text = input_text;
  if (!input_text_file.empty()) {
    bool success = false;
    source_text = ReadTextFile(input_text_file, &success);
    if (!success) {
      return 1;
    }
  }

  // Suppress all log output so only the JSON result is printed.
  litert_lm_set_min_log_level(3);  // 3 = FATAL

  // Create engine settings
  const std::string backend_str = absl::GetFlag(FLAGS_backend);
  LiteRtLmEngineSettings* engine_settings = litert_lm_engine_settings_create(
      model_path.c_str(), backend_str.c_str(),
      /*vision_backend_str=*/nullptr, /*audio_backend_str=*/nullptr);
  if (!engine_settings) {
    std::cerr << "Failed to create engine settings" << std::endl;
    return 1;
  }

  //set dispatch lib dir if provided
  const std::string dispatch_lib_dir =
      absl::GetFlag(FLAGS_litert_dispatch_lib_dir);
  if (!dispatch_lib_dir.empty()) {
    // Note: The C API doesn't expose SetLitertDispatchLibDir yet.
    // This would need to be added if required for NPU support.
    std::cerr << "Warning: --litert_dispatch_lib_dir not supported in C API yet"
              << std::endl;
  }

  // Enable benchmarking
  litert_lm_engine_settings_enable_benchmark(engine_settings);

  // Create engine
  LiteRtLmEngine* engine = litert_lm_engine_create(engine_settings);
  litert_lm_engine_settings_delete(engine_settings);
  if (!engine) {
    std::cerr << "Failed to create engine" << std::endl;
    return 1;
  }

  // Create session config
  LiteRtLmSessionConfig* session_config = litert_lm_session_config_create();
  litert_lm_session_config_set_max_output_tokens(session_config, 2048);

  // Create conversation config
  const bool enable_constrained_decoding = absl::GetFlag(FLAGS_add_constraint);
  LiteRtLmConversationConfig* conversation_config =
      litert_lm_conversation_config_create(
          engine, session_config,
          /*system_message_json=*/nullptr, /*tools_json=*/nullptr,
          /*messages_json=*/nullptr, enable_constrained_decoding);
  litert_lm_session_config_delete(session_config);
  if (!conversation_config) {
    std::cerr << "Failed to create conversation config" << std::endl;
    litert_lm_engine_delete(engine);
    return 1;
  }

  // Create conversation
  LiteRtLmConversation* conversation =
      litert_lm_conversation_create(engine, conversation_config);
  litert_lm_conversation_config_delete(conversation_config);
  if (!conversation) {
    std::cerr << "Failed to create conversation" << std::endl;
    litert_lm_engine_delete(engine);
    return 1;
  }

  // Build the prompt: system instructions + compact schema + source text.
  const std::string compact_schema =
      CompactJsonSchemaForPrompt(kKnowledgeGraphSchema);
  const std::string prompt =
      std::string(kKnowledgeGraphSystemPrompt) + compact_schema +
      "\n\nText to extract from:\n" + source_text +
      "\n\nKnowledge graph JSON:\n";

  // Create message JSON
  json content_list = json::array();
  content_list.push_back({{"type", "text"}, {"text", prompt}});
  json message_json =
      json::object({{"role", "user"}, {"content", content_list}});
  std::string message_str = message_json.dump();

  // Create optional args with constraint if enabled
  LiteRtLmOptionalArgs* optional_args = nullptr;
  if (enable_constrained_decoding) {
    optional_args = litert_lm_optional_args_create();
    litert_lm_optional_args_set_llg_constraint(
        optional_args, kLlgConstraintJsonSchema, kKnowledgeGraphSchema);
  }

  // Send message and get response
  LiteRtLmJsonResponse* response = litert_lm_conversation_send_message_with_args(
      conversation, message_str.c_str(), optional_args);

  if (optional_args) {
    litert_lm_optional_args_delete(optional_args);
  }

  if (!response) {
    std::cerr << "Failed to send message" << std::endl;
    litert_lm_conversation_delete(conversation);
    litert_lm_engine_delete(engine);
    return 1;
  }

  // Get response string and parse
  const char* response_str = litert_lm_json_response_get_string(response);
  if (!response_str) {
    std::cerr << "Failed to get response string" << std::endl;
    litert_lm_json_response_delete(response);
    litert_lm_conversation_delete(conversation);
    litert_lm_engine_delete(engine);
    return 1;
  }

  // Parse the response JSON
  std::string full_response;
  try {
    json response_json = json::parse(response_str);
    if (response_json.contains("content") &&
        response_json["content"].is_array()) {
      for (const auto& content : response_json["content"]) {
        if (content.contains("text")) {
          full_response += content["text"].get<std::string>();
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Failed to parse response: " << e.what() << std::endl;
    litert_lm_json_response_delete(response);
    litert_lm_conversation_delete(conversation);
    litert_lm_engine_delete(engine);
    return 1;
  }

  litert_lm_json_response_delete(response);

  // Try to parse the response as JSON; fall back to brace extraction.
  std::string result_json;
  try {
    json parsed = json::parse(full_response);
    result_json = parsed.dump(2);
  } catch (...) {
    // Model may have generated extra text around the JSON object.
    result_json = ExtractFirstJsonObject(full_response);
    try {
      result_json = json::parse(result_json).dump(2);
    } catch (...) {
      result_json = full_response;
    }
  }

  std::cout << result_json << std::endl;

  // Print benchmark info.
  LiteRtLmBenchmarkInfo* benchmark_info =
      litert_lm_conversation_get_benchmark_info(conversation);
  if (benchmark_info) {
    double ttft = litert_lm_benchmark_info_get_time_to_first_token(
        benchmark_info);
    double init_time =
        litert_lm_benchmark_info_get_total_init_time_in_second(benchmark_info);
    int num_prefill =
        litert_lm_benchmark_info_get_num_prefill_turns(benchmark_info);
    int num_decode =
        litert_lm_benchmark_info_get_num_decode_turns(benchmark_info);

    // Calculate aggregate metrics for benchmark script parsing
    double total_prefill_tokens = 0;
    double total_prefill_time = 0;
    for (int i = 0; i < num_prefill; ++i) {
      int tokens = litert_lm_benchmark_info_get_prefill_token_count_at(
          benchmark_info, i);
      double tps = litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
          benchmark_info, i);
      total_prefill_tokens += tokens;
      if (tps > 0) total_prefill_time += tokens / tps;
    }

    double total_decode_tokens = 0;
    double total_decode_time = 0;
    for (int i = 0; i < num_decode; ++i) {
      int tokens =
          litert_lm_benchmark_info_get_decode_token_count_at(benchmark_info, i);
      double tps = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
          benchmark_info, i);
      total_decode_tokens += tokens;
      if (tps > 0) total_decode_time += tokens / tps;
    }

    double avg_prefill_speed = total_prefill_time > 0 ? total_prefill_tokens / total_prefill_time : 0;
    double avg_decode_speed = total_decode_time > 0 ? total_decode_tokens / total_decode_time : 0;
    double total_time = init_time + total_prefill_time + total_decode_time;

    // Output in format expected by benchmark_android.sh
    std::cerr << "\nPrefill Speed: " << avg_prefill_speed << " tokens/sec\n";
    std::cerr << "Decode Speed: " << avg_decode_speed << " tokens/sec\n";
    std::cerr << "Total Time: " << total_time << " s\n";
    std::cerr << "Time to first token: " << ttft << " s\n";

    litert_lm_benchmark_info_delete(benchmark_info);
  }

  // Cleanup
  litert_lm_conversation_delete(conversation);
  litert_lm_engine_delete(engine);

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  return Run(argc, argv);
}
