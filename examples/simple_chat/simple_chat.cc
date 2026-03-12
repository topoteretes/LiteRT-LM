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

// Knowledge graph extraction demo using LiteRT-LM with constrained decoding.
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
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/status_macros.h"

ABSL_FLAG(std::string, model_path, "", "Path to the .litertlm model file.");
ABSL_FLAG(std::string, backend, "cpu", "Backend to use (cpu, gpu, npu).");
ABSL_FLAG(bool, add_constraint, true, "Enable constrained decoding.");
ABSL_FLAG(std::string, input_text, "", "Input text to extract knowledge from.");
ABSL_FLAG(std::string, input_text_file, "",
          "Path to a file containing the input text.");
ABSL_FLAG(std::string, litert_dispatch_lib_dir, "",
          "Directory of the LiteRT dispatch library (e.g. for NPU/QNN).");

namespace {

using ::litert::lm::Backend;
using ::litert::lm::Conversation;
using ::litert::lm::ConversationConfig;
using ::litert::lm::ConstraintProviderConfig;
using ::litert::lm::Engine;
using ::litert::lm::EngineSettings;
using ::litert::lm::LlGuidanceConfig;
using ::litert::lm::LlGuidanceConstraintArg;
using ::litert::lm::LlgConstraintType;
using ::litert::lm::Message;
using ::litert::lm::ModelAssets;
using ::litert::lm::OptionalArgs;
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
absl::StatusOr<std::string> ReadTextFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return absl::NotFoundError("Could not open file: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
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

absl::Status Run(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("--model_path is required.");
  }
  const std::string input_text = absl::GetFlag(FLAGS_input_text);
  const std::string input_text_file = absl::GetFlag(FLAGS_input_text_file);
  if (input_text.empty() && input_text_file.empty()) {
    return absl::InvalidArgumentError(
        "One of --input_text or --input_text_file is required.");
  }
  if (!input_text.empty() && !input_text_file.empty()) {
    return absl::InvalidArgumentError(
        "Only one of --input_text or --input_text_file may be specified.");
  }

  std::string source_text = input_text;
  if (!input_text_file.empty()) {
    ASSIGN_OR_RETURN(source_text, ReadTextFile(input_text_file));
  }

  // Suppress all log output so only the JSON result is printed.
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kFatal);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kFatal);

  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(model_path));
  ASSIGN_OR_RETURN(Backend backend,
                   litert::lm::GetBackendFromString(absl::GetFlag(FLAGS_backend)));
  ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), backend));

  const std::string dispatch_lib_dir =
      absl::GetFlag(FLAGS_litert_dispatch_lib_dir);
  if (!dispatch_lib_dir.empty()) {
    engine_settings.GetMutableMainExecutorSettings().SetLitertDispatchLibDir(
        dispatch_lib_dir);
  }

  engine_settings.GetMutableBenchmarkParams() =
      litert::lm::proto::BenchmarkParams();

  ASSIGN_OR_RETURN(auto engine,
                   litert::lm::EngineFactory::CreateAny(
                       std::move(engine_settings)));

  auto session_config = litert::lm::SessionConfig::CreateDefault();
  session_config.SetMaxOutputTokens(2048);

  const bool enable_constrained_decoding = absl::GetFlag(FLAGS_add_constraint);
  ASSIGN_OR_RETURN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetEnableConstrainedDecoding(enable_constrained_decoding)
          .SetConstraintProviderConfig(
              ConstraintProviderConfig(LlGuidanceConfig()))
          .Build(*engine));

  ASSIGN_OR_RETURN(auto conversation,
                   Conversation::Create(*engine, conversation_config));

  // Build the prompt: system instructions + compact schema + source text.
  const std::string compact_schema =
      CompactJsonSchemaForPrompt(kKnowledgeGraphSchema);
  const std::string prompt =
      std::string(kKnowledgeGraphSystemPrompt) + compact_schema +
      "\n\nText to extract from:\n" + source_text +
      "\n\nKnowledge graph JSON:\n";

  json content_list = json::array();
  content_list.push_back({{"type", "text"}, {"text", prompt}});

  // Accumulate the full response text.
  std::string full_response;
  auto callback = [&full_response](absl::StatusOr<Message> message) {
    if (!message.ok()) return;
    if (std::holds_alternative<litert::lm::JsonMessage>(*message)) {
      const auto& json_msg = std::get<litert::lm::JsonMessage>(*message);
      if (json_msg.is_null()) return;
      for (const auto& content : json_msg["content"]) {
        if (content.contains("text")) {
          full_response += content["text"].get<std::string>();
        }
      }
    }
  };

  // Attach per-request constraint arg when constrained decoding is enabled.
  OptionalArgs optional_args;
  if (enable_constrained_decoding) {
    optional_args.decoding_constraint = LlGuidanceConstraintArg{
        .constraint_type = LlgConstraintType::kJsonSchema,
        .constraint_string = kKnowledgeGraphSchema,
    };
  }

  RETURN_IF_ERROR(conversation->SendMessageAsync(
      json::object({{"role", "user"}, {"content", content_list}}),
      std::move(callback), std::move(optional_args)));
  RETURN_IF_ERROR(engine->WaitUntilDone(absl::Minutes(10)));

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
  auto benchmark_info = conversation->GetBenchmarkInfo();
  if (benchmark_info.ok()) {
    std::cerr << std::endl << *benchmark_info << std::endl;
  }

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  ABSL_CHECK_OK(Run(argc, argv));
  return 0;
}
