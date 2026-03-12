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

#include "runtime/components/constrained_decoding/constrained_decoder.h"

#include <cstdint>
#include <limits>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  //NOLINT
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

absl::Status ConstrainedDecoder::UpdateConstraintState(
    const ::litert::TensorBuffer& next_token_ids) {
  LITERT_ASSIGN_OR_RETURN(auto next_token_ids_span,
                          ReferTensorBufferAsSpan<int>(next_token_ids));
  return UpdateConstraintState(next_token_ids_span);
}

absl::Status ConstrainedDecoder::UpdateConstraintState(
    absl::Span<int> next_token_ids) {
  RET_CHECK_EQ(next_token_ids.size(), batch_size_)
      << "Batch size [" << next_token_ids.size()
      << "] does not match the expected batch size [" << batch_size_ << "].";
  for (int i = 0; i < batch_size_; ++i) {
    auto& constraint_state = constraint_states_[i];
    ASSIGN_OR_RETURN(
        constraint_state,
        constraint_->ComputeNext(*constraint_state, next_token_ids[i]));
    if (constraint_->IsEnded(*constraint_state)) {
      constraint_state = constraint_->Start();
    }
  }
  return absl::OkStatus();
}

absl::Status ConstrainedDecoder::MaskLogits(::litert::TensorBuffer& logits) {
  // Compute the allowed tokens bitmap for the current constraint state.
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
  if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto logits_span,
                            ReferTensorBufferAsSpan<float>(logits));
    return MaskLogits(logits_span, logits_tensor_type.Layout().Dimensions());
  } else if (logits_tensor_type.ElementType() ==
             ::litert::ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(auto logits_span,
                            ReferTensorBufferAsSpan<tflite::half>(logits));
    return MaskLogits(logits_span, logits_tensor_type.Layout().Dimensions());
  }
  return absl::InvalidArgumentError("Unsupported logits type for MaskLogits.");
}

absl::Status ConstrainedDecoder::MaskLogits(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  RET_CHECK_EQ(logits_dims.size(), 3)
      << "Only support logits with dimensions [batch_size, 1, vocab_size].";
  int batch_size = logits_dims[0];
  int sequence_length = logits_dims[1];
  int vocab_size = logits_dims[2];
  RET_CHECK_EQ(sequence_length, 1) << "Only support sequence length 1.";
  // It is possible that the constraint vocabulary size is larger than the model
  // vocabulary size. The remaining tokens in the constraint vocabulary are
  // treated as unused tokens.
  RET_CHECK_LE(vocab_size, constraint_->GetVocabularySize())
      << "Vocabulary size [" << vocab_size
      << "] does not match the expected vocabulary size ["
      << constraint_->GetVocabularySize() << "].";
  RET_CHECK_EQ(batch_size, batch_size_)
      << "Batch size [" << batch_size
      << "] does not match the expected batch size [" << batch_size_ << "].";
  for (int b = 0; b < batch_size; ++b) {
    auto& constraint_state = constraint_states_[b];
    ASSIGN_OR_RETURN(auto bitmap,
                     constraint_->ComputeBitmap(*constraint_state));
    for (int i = 0; i < vocab_size; ++i) {
      if (!bitmap->Get(i)) {
        logits.data()[b * vocab_size + i] =
            std::numeric_limits<float>::lowest();
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConstrainedDecoder::MaskLogits(
    absl::Span<tflite::half> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  RET_CHECK_EQ(logits_dims.size(), 3)
      << "Only support logits with dimensions [batch_size, 1, vocab_size].";
  int batch_size = logits_dims[0];
  int sequence_length = logits_dims[1];
  int vocab_size = logits_dims[2];
  RET_CHECK_EQ(sequence_length, 1) << "Only support sequence length 1.";
  // It is possible that the constraint vocabulary size is larger than the model
  // vocabulary size. The remaining tokens in the constraint vocabulary are
  // treated as unused tokens.
  RET_CHECK_LE(vocab_size, constraint_->GetVocabularySize())
      << "Vocabulary size [" << vocab_size
      << "] does not match the expected vocabulary size ["
      << constraint_->GetVocabularySize() << "].";
  RET_CHECK_EQ(batch_size, batch_size_)
      << "Batch size [" << batch_size
      << "] does not match the expected batch size [" << batch_size_ << "].";
  for (int b = 0; b < batch_size; ++b) {
    auto& constraint_state = constraint_states_[b];
    ASSIGN_OR_RETURN(auto bitmap,
                     constraint_->ComputeBitmap(*constraint_state));
    for (int i = 0; i < vocab_size; ++i) {
      if (!bitmap->Get(i)) {
        logits.data()[b * vocab_size + i] = tflite::half::min();
      }
    }
  }
  return absl::OkStatus();
}

#ifdef LITERT_LM_ASYNC_CONSTRAINT_MASKING

absl::Status ConstrainedDecoder::StartPrecomputeMask(ThreadPool& thread_pool) {
  {
    absl::MutexLock lock(&mask_mutex_);
    mask_ready_ = false;
    mask_status_ = absl::OkStatus();
    precomputed_bitmaps_.clear();
    precomputed_bitmaps_.resize(batch_size_);
  }

  // Capture raw pointers to constraint states — they won't change until
  // ApplyPrecomputedMask is called (which waits for this to complete).
  return thread_pool.Schedule(
      [this]() {
        absl::Status status = absl::OkStatus();
        std::vector<std::unique_ptr<Bitmap>> bitmaps(batch_size_);
        for (int b = 0; b < batch_size_; ++b) {
          auto bitmap_or = constraint_->ComputeBitmap(*constraint_states_[b]);
          if (!bitmap_or.ok()) {
            status = bitmap_or.status();
            break;
          }
          bitmaps[b] = std::move(*bitmap_or);
        }
        absl::MutexLock lock(&mask_mutex_);
        precomputed_bitmaps_ = std::move(bitmaps);
        mask_status_ = status;
        mask_ready_ = true;
      });
}

absl::Status ConstrainedDecoder::ApplyPrecomputedMask(
    ::litert::TensorBuffer& logits) {
  LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto logits_span, ReferTensorBufferAsSpan<float>(logits));
  return ApplyPrecomputedMask(logits_span,
                              logits_tensor_type.Layout().Dimensions());
}

absl::Status ConstrainedDecoder::ApplyPrecomputedMask(
    absl::Span<float> logits,
    absl::Span<const ::litert::Layout::Dim> logits_dims) {
  RET_CHECK_EQ(logits_dims.size(), 3)
      << "Only support logits with dimensions [batch_size, 1, vocab_size].";
  int batch_size = logits_dims[0];
  int sequence_length = logits_dims[1];
  int vocab_size = logits_dims[2];
  RET_CHECK_EQ(sequence_length, 1) << "Only support sequence length 1.";
  RET_CHECK_LE(vocab_size, constraint_->GetVocabularySize())
      << "Vocabulary size [" << vocab_size
      << "] does not match the expected vocabulary size ["
      << constraint_->GetVocabularySize() << "].";
  RET_CHECK_EQ(batch_size, batch_size_)
      << "Batch size [" << batch_size
      << "] does not match the expected batch size [" << batch_size_ << "].";

  // Wait for the precomputed bitmaps to be ready.
  {
    absl::MutexLock lock(&mask_mutex_);
    mask_mutex_.Await(absl::Condition(&mask_ready_));
    RETURN_IF_ERROR(mask_status_);
  }

  // Apply the precomputed bitmaps to the logits.
  for (int b = 0; b < batch_size; ++b) {
    const auto& bitmap = precomputed_bitmaps_[b];
    for (int i = 0; i < vocab_size; ++i) {
      if (!bitmap->Get(i)) {
        logits.data()[b * vocab_size + i] =
            std::numeric_limits<float>::lowest();
      }
    }
  }

  // Reset for next step.
  {
    absl::MutexLock lock(&mask_mutex_);
    mask_ready_ = false;
    precomputed_bitmaps_.clear();
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> ConstrainedDecoder::ValidateSpeculativeTokens(
    absl::Span<const int> token_ids) {
  RET_CHECK_EQ(static_cast<int>(token_ids.size()), batch_size_)
      << "Token IDs size [" << token_ids.size()
      << "] does not match batch size [" << batch_size_ << "].";

  // Wait for the precomputed bitmaps to be ready.
  {
    absl::MutexLock lock(&mask_mutex_);
    mask_mutex_.Await(absl::Condition(&mask_ready_));
    RETURN_IF_ERROR(mask_status_);
  }

  // Check each token against its bitmap.
  for (int b = 0; b < batch_size_; ++b) {
    if (!precomputed_bitmaps_[b]->Get(token_ids[b])) {
      return false;
    }
  }
  return true;
}

absl::Status ConstrainedDecoder::AcceptSpeculativeTokens(
    absl::Span<const int> token_ids) {
  RET_CHECK_EQ(static_cast<int>(token_ids.size()), batch_size_)
      << "Token IDs size [" << token_ids.size()
      << "] does not match batch size [" << batch_size_ << "].";

  // Advance constraint state for each batch element.
  for (int i = 0; i < batch_size_; ++i) {
    auto& constraint_state = constraint_states_[i];
    ASSIGN_OR_RETURN(
        constraint_state,
        constraint_->ComputeNext(*constraint_state, token_ids[i]));
    if (constraint_->IsEnded(*constraint_state)) {
      constraint_state = constraint_->Start();
    }
  }

  // Clear precomputed bitmaps.
  {
    absl::MutexLock lock(&mask_mutex_);
    mask_ready_ = false;
    precomputed_bitmaps_.clear();
  }
  return absl::OkStatus();
}

#endif  // LITERT_LM_ASYNC_CONSTRAINT_MASKING

}  // namespace litert::lm
