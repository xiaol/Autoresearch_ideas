//
// Created by dengzi on 2025/7/14.
//

#ifndef LLAMA_CPP_EMBEDDING_H
#define LLAMA_CPP_EMBEDDING_H

#include "llama.h"
#include <string>
#include <vector>

class rwkv_embedding {
public:
    explicit rwkv_embedding();

    int load_model(const std::string &model_path);

    int load_rerank_model(const std::string &model_path);

    std::vector<std::vector<float> > get_embeddings(const std::vector<std::string> &inputs) const;

    std::vector<float> rerank(const std::string &query, const std::vector<std::string> &documents) const;

    void release();

    static float similarity(const std::vector<float> &emb1, const std::vector<float> &emb2);

private:
    llama_model *model;
    llama_context *ctx;
    int embd_normalize_type = 2;

    llama_model *model_rerank;
    llama_context *ctx_rerank;

    static void embd_normalize(const float *inp, float *out, int n, int normalization);

    static int batch_add_seq(llama_batch &batch, const std::vector<int32_t> &tokens, llama_seq_id seq_id);

    static int batch_decode(llama_context *ctx, const llama_batch &batch, float *output, int n_embd, int normalization);

    float rank_document(const std::string &query, const std::string &document) const;

    static std::vector<int32_t> tokenize(const llama_model *model, const std::string &text, bool add_special,
                                         bool parse_special, bool add_eos, bool add_sep);
};

#endif // LLAMA_CPP_EMBEDDING_H
