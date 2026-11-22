"""Export and execute a native C99 runtime for trained module-I/O checkpoints.

FP components (token/position embeddings, LayerNorm, attention Q/K/V/softmax/out,
vocabulary head, ToLogic/FromLogic bridges) run as exact C99 float32 with a causal
KV-cache. Discrete Griffith LutLayer stages run as bit-level Boolean truth-table
lookups (compiled directly from ``truth_tables_from_lut_layer`` and argmax-resolved
``LearnableMapping`` indices); layer 0 of ``LogicMLP`` supports both exact continuous
Walsh-Hadamard evaluation (matching PyTorch ``hard`` mode bitwise-close) and
binarized pure-Boolean truth-table evaluation.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import torch

from griffith_logic.export.discretize import truth_tables_from_lut_layer

from .model import (
    BOS,
    BYTE_OFFSET,
    EOS,
    PAD,
    SEP,
    VOCAB_SIZE,
    ExampleTooLongError,
    ModelConfig,
    encode_text,
    load_checkpoint,
)


_MAGIC = 0x4D494F43  # 'MIOC'
_FORMAT_VERSION = 1

_RUNTIME_H = r"""#ifndef MODULEIO_RUNTIME_H
#define MODULEIO_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ModuleIOCModel ModuleIOCModel;

ModuleIOCModel *moduleio_c_load(const char *weights_path);
void moduleio_c_free(ModuleIOCModel *model);
int moduleio_c_info(const ModuleIOCModel *model, int32_t *out_info8);

int moduleio_c_forward_last(
    ModuleIOCModel *model,
    const int32_t *token_ids,
    int32_t seq_len,
    int32_t binarize_tologic,
    float *out_logits);

int moduleio_c_generate(
    ModuleIOCModel *model,
    const int32_t *prompt_ids,
    int32_t prompt_len,
    int32_t max_new_tokens,
    int32_t use_kv_cache,
    int32_t binarize_tologic,
    int32_t *out_tokens,
    int32_t *out_len,
    int32_t *out_eos,
    uint64_t *out_profile_ns7);

#ifdef __cplusplus
}
#endif

#endif
"""

_RUNTIME_C = r"""#define _POSIX_C_SOURCE 200809L
#include "moduleio_runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIOC_MAGIC 0x4D494F43
#define MIOC_VERSION 1
#define PAD_ID 0
#define BOS_ID 1
#define EOS_ID 2
#define SEP_ID 3

typedef struct {
  float *norm1_w;
  float *norm1_b;
  float *in_proj_w;
  float *in_proj_b;
  float *out_proj_w;
  float *out_proj_b;
  float *norm2_w;
  float *norm2_b;
  /* Logic FFN weights */
  float inv_tau;
  float from_tau;
  uint16_t *lut0_idx;
  float *lut0_w;
  uint64_t *lut0_tt;
  uint16_t *lut1_idx;
  uint64_t *lut1_tt;
  /* FP FFN weights */
  float *fc1_w;
  float *fc1_b;
  float *fc2_w;
  float *fc2_b;
} LayerWeights;

struct ModuleIOCModel {
  int32_t d_model;
  int32_t n_heads;
  int32_t n_layers;
  int32_t mlp_ratio;
  int32_t vocab_size;
  int32_t max_seq_len;
  int32_t ffn_is_logic;
  int32_t ffn_n_inputs;
  int32_t ffn_width;
  int32_t head_dim;
  float inv_sqrt_head_dim;

  float *token_embed;
  float *pos_embed;
  LayerWeights *layers;
  float *norm_w;
  float *norm_b;
  float *head_w;
  float *head_b;

  /* Scratch buffers */
  float *x;
  float *x_norm;
  float *q;
  float *k;
  float *v;
  float *ctx;
  float *attn_out;
  float *scores;
  float *ffn_sig;
  float *ffn_fp_hidden;
  uint8_t *lut_h1;
  uint8_t *lut_h2;
  float *kv_k;
  float *kv_v;
};

static inline uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int read_exact(FILE *fp, void *dst, size_t nbytes) {
  if (nbytes == 0) return 0;
  return fread(dst, 1, nbytes, fp) == nbytes ? 0 : -1;
}

static void *alloc_read(FILE *fp, size_t count, size_t elem_size) {
  size_t total = count * elem_size;
  void *ptr = malloc(total > 0 ? total : 1);
  if (!ptr) return NULL;
  if (read_exact(fp, ptr, total) != 0) {
    free(ptr);
    return NULL;
  }
  return ptr;
}

void moduleio_c_free(ModuleIOCModel *m) {
  int32_t l;
  if (!m) return;
  free(m->token_embed);
  free(m->pos_embed);
  if (m->layers) {
    for (l = 0; l < m->n_layers; l++) {
      free(m->layers[l].norm1_w);
      free(m->layers[l].norm1_b);
      free(m->layers[l].in_proj_w);
      free(m->layers[l].in_proj_b);
      free(m->layers[l].out_proj_w);
      free(m->layers[l].out_proj_b);
      free(m->layers[l].norm2_w);
      free(m->layers[l].norm2_b);
      free(m->layers[l].lut0_idx);
      free(m->layers[l].lut0_w);
      free(m->layers[l].lut0_tt);
      free(m->layers[l].lut1_idx);
      free(m->layers[l].lut1_tt);
      free(m->layers[l].fc1_w);
      free(m->layers[l].fc1_b);
      free(m->layers[l].fc2_w);
      free(m->layers[l].fc2_b);
    }
    free(m->layers);
  }
  free(m->norm_w);
  free(m->norm_b);
  free(m->head_w);
  free(m->head_b);
  free(m->x);
  free(m->x_norm);
  free(m->q);
  free(m->k);
  free(m->v);
  free(m->ctx);
  free(m->attn_out);
  free(m->scores);
  free(m->ffn_sig);
  free(m->ffn_fp_hidden);
  free(m->lut_h1);
  free(m->lut_h2);
  free(m->kv_k);
  free(m->kv_v);
  free(m);
}

ModuleIOCModel *moduleio_c_load(const char *weights_path) {
  FILE *fp;
  int32_t hdr[16];
  ModuleIOCModel *m;
  int32_t l;
  size_t D, W, V, S, L;

  if (!weights_path) return NULL;
  fp = fopen(weights_path, "rb");
  if (!fp) return NULL;
  if (read_exact(fp, hdr, sizeof(hdr)) != 0 ||
      hdr[0] != MIOC_MAGIC || hdr[1] != MIOC_VERSION) {
    fclose(fp);
    return NULL;
  }

  m = (ModuleIOCModel *)calloc(1, sizeof(ModuleIOCModel));
  if (!m) {
    fclose(fp);
    return NULL;
  }
  m->d_model = hdr[2];
  m->n_heads = hdr[3];
  m->n_layers = hdr[4];
  m->mlp_ratio = hdr[5];
  m->vocab_size = hdr[6];
  m->max_seq_len = hdr[7];
  m->ffn_is_logic = hdr[8];
  m->ffn_n_inputs = hdr[9];

  if (m->d_model <= 0 || m->n_heads <= 0 || m->d_model % m->n_heads != 0 ||
      m->n_layers <= 0 || m->mlp_ratio <= 0 || m->vocab_size <= 0 ||
      m->max_seq_len <= 0) {
    moduleio_c_free(m);
    fclose(fp);
    return NULL;
  }
  if (m->ffn_is_logic &&
      m->ffn_n_inputs != 2 && m->ffn_n_inputs != 4 && m->ffn_n_inputs != 6) {
    moduleio_c_free(m);
    fclose(fp);
    return NULL;
  }

  m->ffn_width = m->d_model * m->mlp_ratio;
  m->head_dim = m->d_model / m->n_heads;
  m->inv_sqrt_head_dim = 1.0f / sqrtf((float)m->head_dim);

  D = (size_t)m->d_model;
  W = (size_t)m->ffn_width;
  V = (size_t)m->vocab_size;
  S = (size_t)m->max_seq_len;
  L = (size_t)m->n_layers;

  m->token_embed = (float *)alloc_read(fp, V * D, sizeof(float));
  m->pos_embed = (float *)alloc_read(fp, S * D, sizeof(float));
  m->layers = (LayerWeights *)calloc(L, sizeof(LayerWeights));
  if (!m->token_embed || !m->pos_embed || !m->layers) {
    moduleio_c_free(m);
    fclose(fp);
    return NULL;
  }

  for (l = 0; l < m->n_layers; l++) {
    LayerWeights *lw = &m->layers[l];
    lw->norm1_w = (float *)alloc_read(fp, D, sizeof(float));
    lw->norm1_b = (float *)alloc_read(fp, D, sizeof(float));
    lw->in_proj_w = (float *)alloc_read(fp, 3 * D * D, sizeof(float));
    lw->in_proj_b = (float *)alloc_read(fp, 3 * D, sizeof(float));
    lw->out_proj_w = (float *)alloc_read(fp, D * D, sizeof(float));
    lw->out_proj_b = (float *)alloc_read(fp, D, sizeof(float));
    lw->norm2_w = (float *)alloc_read(fp, D, sizeof(float));
    lw->norm2_b = (float *)alloc_read(fp, D, sizeof(float));
    if (!lw->norm1_w || !lw->norm1_b || !lw->in_proj_w || !lw->in_proj_b ||
        !lw->out_proj_w || !lw->out_proj_b || !lw->norm2_w || !lw->norm2_b) {
      moduleio_c_free(m);
      fclose(fp);
      return NULL;
    }
    if (m->ffn_is_logic) {
      size_t nin = (size_t)m->ffn_n_inputs;
      size_t nent = (size_t)1 << nin;
      float taus[2];
      if (read_exact(fp, taus, sizeof(taus)) != 0) {
        moduleio_c_free(m);
        fclose(fp);
        return NULL;
      }
      lw->inv_tau = taus[0];
      lw->from_tau = taus[1];
      lw->lut0_idx = (uint16_t *)alloc_read(fp, W * nin, sizeof(uint16_t));
      lw->lut0_w = (float *)alloc_read(fp, W * nent, sizeof(float));
      lw->lut0_tt = (uint64_t *)alloc_read(fp, W, sizeof(uint64_t));
      lw->lut1_idx = (uint16_t *)alloc_read(fp, W * nin, sizeof(uint16_t));
      lw->lut1_tt = (uint64_t *)alloc_read(fp, W, sizeof(uint64_t));
      if (!lw->lut0_idx || !lw->lut0_w || !lw->lut0_tt ||
          !lw->lut1_idx || !lw->lut1_tt) {
        moduleio_c_free(m);
        fclose(fp);
        return NULL;
      }
    } else {
      lw->fc1_w = (float *)alloc_read(fp, W * D, sizeof(float));
      lw->fc1_b = (float *)alloc_read(fp, W, sizeof(float));
      lw->fc2_w = (float *)alloc_read(fp, D * W, sizeof(float));
      lw->fc2_b = (float *)alloc_read(fp, D, sizeof(float));
      if (!lw->fc1_w || !lw->fc1_b || !lw->fc2_w || !lw->fc2_b) {
        moduleio_c_free(m);
        fclose(fp);
        return NULL;
      }
    }
  }

  m->norm_w = (float *)alloc_read(fp, D, sizeof(float));
  m->norm_b = (float *)alloc_read(fp, D, sizeof(float));
  m->head_w = (float *)alloc_read(fp, V * D, sizeof(float));
  m->head_b = (float *)alloc_read(fp, V, sizeof(float));
  fclose(fp);

  if (!m->norm_w || !m->norm_b || !m->head_w || !m->head_b) {
    moduleio_c_free(m);
    return NULL;
  }

  m->x = (float *)malloc(D * sizeof(float));
  m->x_norm = (float *)malloc(D * sizeof(float));
  m->q = (float *)malloc(D * sizeof(float));
  m->k = (float *)malloc(D * sizeof(float));
  m->v = (float *)malloc(D * sizeof(float));
  m->ctx = (float *)malloc(D * sizeof(float));
  m->attn_out = (float *)malloc(D * sizeof(float));
  m->scores = (float *)malloc(S * sizeof(float));
  m->ffn_sig = (float *)malloc(D * sizeof(float));
  m->ffn_fp_hidden = (float *)malloc(W * sizeof(float));
  m->lut_h1 = (uint8_t *)malloc(W * sizeof(uint8_t));
  m->lut_h2 = (uint8_t *)malloc(W * sizeof(uint8_t));
  m->kv_k = (float *)malloc(L * S * D * sizeof(float));
  m->kv_v = (float *)malloc(L * S * D * sizeof(float));

  if (!m->x || !m->x_norm || !m->q || !m->k || !m->v || !m->ctx ||
      !m->attn_out || !m->scores || !m->ffn_sig || !m->ffn_fp_hidden ||
      !m->lut_h1 || !m->lut_h2 || !m->kv_k || !m->kv_v) {
    moduleio_c_free(m);
    return NULL;
  }
  return m;
}

int moduleio_c_info(const ModuleIOCModel *m, int32_t *out_info8) {
  if (!m || !out_info8) return -1;
  out_info8[0] = m->d_model;
  out_info8[1] = m->n_heads;
  out_info8[2] = m->n_layers;
  out_info8[3] = m->mlp_ratio;
  out_info8[4] = m->vocab_size;
  out_info8[5] = m->max_seq_len;
  out_info8[6] = m->ffn_is_logic;
  out_info8[7] = m->ffn_n_inputs;
  return 0;
}

static inline void layer_norm(
    const float *x, const float *w, const float *b, float *y, int32_t D) {
  int32_t d;
  float sum = 0.0f;
  float var = 0.0f;
  float mean, inv_std;
  for (d = 0; d < D; d++) sum += x[d];
  mean = sum / (float)D;
  for (d = 0; d < D; d++) {
    float diff = x[d] - mean;
    var += diff * diff;
  }
  inv_std = 1.0f / sqrtf(var / (float)D + 1e-5f);
  for (d = 0; d < D; d++) {
    y[d] = (x[d] - mean) * inv_std * w[d] + b[d];
  }
}

static inline void matvec_add_bias(
    const float *W, const float *b, const float *x, float *y, int32_t rows, int32_t cols) {
  int32_t r, c;
  for (r = 0; r < rows; r++) {
    const float *wr = W + (size_t)r * cols;
    float acc = b[r];
    for (c = 0; c < cols; c++) acc += wr[c] * x[c];
    y[r] = acc;
  }
}

static void lut_wh_layer_n(
    const float *s_in,
    const uint16_t *idx,
    const float *w,
    uint8_t *out_bits,
    int32_t width,
    int32_t n_inputs) {
  int32_t o, j, step, m;
  float basis[64];
  if (n_inputs == 2) {
    for (o = 0; o < width; o++) {
      const uint16_t *io = idx + o * 2;
      const float *wo = w + o * 4;
      float s0 = s_in[io[0]];
      float s1 = s_in[io[1]];
      float logit = wo[0] + wo[1] * s0 + wo[2] * s1 + wo[3] * (s0 * s1);
      out_bits[o] = (logit > 0.0f) ? 1u : 0u;
    }
    return;
  }
  if (n_inputs == 4) {
    for (o = 0; o < width; o++) {
      const uint16_t *io = idx + o * 4;
      const float *wo = w + o * 16;
      float s0 = s_in[io[0]];
      float s1 = s_in[io[1]];
      float s2 = s_in[io[2]];
      float s3 = s_in[io[3]];
      float b0 = 1.0f, b1 = s0;
      float b2 = s1, b3 = b1 * s1;
      float b4 = b0 * s2, b5 = b1 * s2, b6 = b2 * s2, b7 = b3 * s2;
      float b8 = b0 * s3, b9 = b1 * s3, b10 = b2 * s3, b11 = b3 * s3;
      float b12 = b4 * s3, b13 = b5 * s3, b14 = b6 * s3, b15 = b7 * s3;
      float logit = wo[0] * b0 + wo[1] * b1 + wo[2] * b2 + wo[3] * b3 +
                    wo[4] * b4 + wo[5] * b5 + wo[6] * b6 + wo[7] * b7 +
                    wo[8] * b8 + wo[9] * b9 + wo[10] * b10 + wo[11] * b11 +
                    wo[12] * b12 + wo[13] * b13 + wo[14] * b14 + wo[15] * b15;
      out_bits[o] = (logit > 0.0f) ? 1u : 0u;
    }
    return;
  }
  basis[0] = 1.0f;
  for (o = 0; o < width; o++) {
    const uint16_t *io = idx + (size_t)o * n_inputs;
    const int32_t nent = 1 << n_inputs;
    const float *wo = w + (size_t)o * nent;
    float logit = 0.0f;
    for (j = 0; j < n_inputs; j++) {
      float sj = s_in[io[j]];
      step = 1 << j;
      for (m = 0; m < step; m++) basis[m + step] = basis[m] * sj;
    }
    for (m = 0; m < nent; m++) logit += wo[m] * basis[m];
    out_bits[o] = (logit > 0.0f) ? 1u : 0u;
  }
}

static inline void lut_table_lookup(
    const uint8_t *in_bits,
    const uint16_t *idx,
    const uint64_t *tt,
    uint8_t *out_bits,
    int32_t width,
    int32_t n_inputs) {
  int32_t o;
  if (n_inputs == 2) {
    for (o = 0; o < width; o++) {
      const uint16_t *io = idx + o * 2;
      unsigned addr = (unsigned)in_bits[io[0]] | ((unsigned)in_bits[io[1]] << 1);
      out_bits[o] = (uint8_t)((tt[o] >> addr) & 1ULL);
    }
    return;
  }
  if (n_inputs == 4) {
    for (o = 0; o < width; o++) {
      const uint16_t *io = idx + o * 4;
      unsigned addr = (unsigned)in_bits[io[0]] |
                      ((unsigned)in_bits[io[1]] << 1) |
                      ((unsigned)in_bits[io[2]] << 2) |
                      ((unsigned)in_bits[io[3]] << 3);
      out_bits[o] = (uint8_t)((tt[o] >> addr) & 1ULL);
    }
    return;
  }
  for (o = 0; o < width; o++) {
    const uint16_t *io = idx + (size_t)o * 6;
    unsigned addr = (unsigned)in_bits[io[0]] |
                    ((unsigned)in_bits[io[1]] << 1) |
                    ((unsigned)in_bits[io[2]] << 2) |
                    ((unsigned)in_bits[io[3]] << 3) |
                    ((unsigned)in_bits[io[4]] << 4) |
                    ((unsigned)in_bits[io[5]] << 5);
    out_bits[o] = (uint8_t)((tt[o] >> addr) & 1ULL);
  }
}

static void step_token(
    ModuleIOCModel *m,
    int32_t pos,
    int32_t token_id,
    int32_t binarize_tologic,
    float *out_logits,
    uint64_t *prof_ns) {
  const int32_t D = m->d_model;
  const int32_t W = m->ffn_width;
  const int32_t H = m->n_heads;
  const int32_t HD = m->head_dim;
  const size_t cache_stride = (size_t)m->max_seq_len * (size_t)D;
  const float *te = m->token_embed + (size_t)token_id * (size_t)D;
  const float *pe = m->pos_embed + (size_t)pos * (size_t)D;
  int32_t d, l, h, t, i;
  uint64_t t0 = 0, t1 = 0;

  for (d = 0; d < D; d++) m->x[d] = te[d] + pe[d];

  for (l = 0; l < m->n_layers; l++) {
    const LayerWeights *lw = &m->layers[l];
    float *k_layer = m->kv_k + (size_t)l * cache_stride;
    float *v_layer = m->kv_v + (size_t)l * cache_stride;
    float *k_cur = k_layer + (size_t)pos * (size_t)D;
    float *v_cur = v_layer + (size_t)pos * (size_t)D;

    if (prof_ns) t0 = now_ns();
    layer_norm(m->x, lw->norm1_w, lw->norm1_b, m->x_norm, D);
    matvec_add_bias(lw->in_proj_w, lw->in_proj_b, m->x_norm, m->q, D, D);
    matvec_add_bias(lw->in_proj_w + (size_t)D * D, lw->in_proj_b + D, m->x_norm, k_cur, D, D);
    matvec_add_bias(lw->in_proj_w + (size_t)2 * D * D, lw->in_proj_b + 2 * D, m->x_norm, v_cur, D, D);

    for (h = 0; h < H; h++) {
      const int32_t off = h * HD;
      const float *qh = m->q + off;
      float *ch = m->ctx + off;
      float max_s = -INFINITY;
      float sum_e = 0.0f;

      for (t = 0; t <= pos; t++) {
        const float *kt = k_layer + (size_t)t * (size_t)D + off;
        float dot = 0.0f;
        for (i = 0; i < HD; i++) dot += qh[i] * kt[i];
        dot *= m->inv_sqrt_head_dim;
        m->scores[t] = dot;
        if (dot > max_s) max_s = dot;
      }
      for (t = 0; t <= pos; t++) {
        float e = expf(m->scores[t] - max_s);
        m->scores[t] = e;
        sum_e += e;
      }
      {
        const float inv_sum = 1.0f / sum_e;
        for (i = 0; i < HD; i++) ch[i] = 0.0f;
        for (t = 0; t <= pos; t++) {
          const float prob = m->scores[t] * inv_sum;
          const float *vt = v_layer + (size_t)t * (size_t)D + off;
          for (i = 0; i < HD; i++) ch[i] += prob * vt[i];
        }
      }
    }
    matvec_add_bias(lw->out_proj_w, lw->out_proj_b, m->ctx, m->attn_out, D, D);
    for (d = 0; d < D; d++) m->x[d] += m->attn_out[d];
    if (prof_ns) {
      t1 = now_ns();
      prof_ns[3] += t1 - t0;
      t0 = t1;
    }

    layer_norm(m->x, lw->norm2_w, lw->norm2_b, m->x_norm, D);
    if (m->ffn_is_logic) {
      const int32_t G = W / D;
      const float half_g = 0.5f * (float)G;
      const float inv_from_tau = 1.0f / lw->from_tau;
      if (binarize_tologic) {
        uint8_t *in_bits = (uint8_t *)m->ffn_sig;
        for (d = 0; d < D; d++) in_bits[d] = (m->x_norm[d] > 0.0f) ? 1u : 0u;
        lut_table_lookup(in_bits, lw->lut0_idx, lw->lut0_tt, m->lut_h1, W, m->ffn_n_inputs);
      } else {
        for (d = 0; d < D; d++) {
          float sig = 1.0f / (1.0f + expf(-m->x_norm[d] * lw->inv_tau));
          m->ffn_sig[d] = 1.0f - 2.0f * sig;
        }
        lut_wh_layer_n(m->ffn_sig, lw->lut0_idx, lw->lut0_w, m->lut_h1, W, m->ffn_n_inputs);
      }
      if (prof_ns) {
        t1 = now_ns();
        prof_ns[4] += t1 - t0;
        t0 = t1;
      }
      lut_table_lookup(m->lut_h1, lw->lut1_idx, lw->lut1_tt, m->lut_h2, W, m->ffn_n_inputs);
      for (d = 0; d < D; d++) {
        const uint8_t *grp = m->lut_h2 + d * G;
        int32_t gsum = 0;
        for (i = 0; i < G; i++) gsum += (int32_t)grp[i];
        m->x[d] += ((float)gsum - half_g) * inv_from_tau;
      }
      if (prof_ns) {
        t1 = now_ns();
        prof_ns[5] += t1 - t0;
      }
    } else {
      matvec_add_bias(lw->fc1_w, lw->fc1_b, m->x_norm, m->ffn_fp_hidden, W, D);
      for (i = 0; i < W; i++) {
        float v = m->ffn_fp_hidden[i];
        m->ffn_fp_hidden[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
      }
      matvec_add_bias(lw->fc2_w, lw->fc2_b, m->ffn_fp_hidden, m->attn_out, D, W);
      for (d = 0; d < D; d++) m->x[d] += m->attn_out[d];
      if (prof_ns) {
        t1 = now_ns();
        prof_ns[4] += t1 - t0;
      }
    }
  }

  if (out_logits) {
    if (prof_ns) t0 = now_ns();
    layer_norm(m->x, m->norm_w, m->norm_b, m->x_norm, D);
    matvec_add_bias(m->head_w, m->head_b, m->x_norm, out_logits, m->vocab_size, D);
    if (prof_ns) prof_ns[6] += now_ns() - t0;
  }
}

int moduleio_c_forward_last(
    ModuleIOCModel *m,
    const int32_t *token_ids,
    int32_t seq_len,
    int32_t binarize_tologic,
    float *out_logits) {
  int32_t pos;
  if (!m || !token_ids || !out_logits || seq_len < 1 || seq_len > m->max_seq_len) {
    return -1;
  }
  for (pos = 0; pos < seq_len; pos++) {
    int32_t tid = token_ids[pos];
    if (tid < 0 || tid >= m->vocab_size) return -1;
    step_token(
        m, pos, tid, binarize_tologic,
        (pos == seq_len - 1) ? out_logits : NULL,
        NULL);
  }
  return 0;
}

static int32_t pick_next_token(float *logits, int32_t vocab_size) {
  int32_t v;
  int32_t best_id = 0;
  float best_val;
  logits[PAD_ID] = -INFINITY;
  logits[BOS_ID] = -INFINITY;
  logits[SEP_ID] = -INFINITY;
  best_val = logits[0];
  for (v = 1; v < vocab_size; v++) {
    if (logits[v] > best_val) {
      best_val = logits[v];
      best_id = v;
    }
  }
  return best_id;
}

int moduleio_c_generate(
    ModuleIOCModel *m,
    const int32_t *prompt_ids,
    int32_t prompt_len,
    int32_t max_new_tokens,
    int32_t use_kv_cache,
    int32_t binarize_tologic,
    int32_t *out_tokens,
    int32_t *out_len,
    int32_t *out_eos,
    uint64_t *out_profile_ns7) {
  const int32_t V = m ? m->vocab_size : 0;
  float *logits;
  int32_t *seq_buf = NULL;
  int32_t pos, step;
  int32_t count = 0;
  int32_t eos_hit = 0;
  uint64_t prof[7];
  uint64_t t_start = 0, t_after_prefill = 0, t_end = 0;

  if (!m || !prompt_ids || !out_len || !out_eos || prompt_len < 1 ||
      max_new_tokens < 0 || prompt_len + max_new_tokens > m->max_seq_len) {
    return -1;
  }
  for (pos = 0; pos < prompt_len; pos++) {
    if (prompt_ids[pos] < 0 || prompt_ids[pos] >= V) return -1;
  }

  memset(prof, 0, sizeof(prof));
  uint64_t *step_prof = out_profile_ns7 ? prof : NULL;
  logits = (float *)malloc((size_t)V * sizeof(float));
  if (!logits) return -1;

  if (!use_kv_cache) {
    seq_buf = (int32_t *)malloc((size_t)(prompt_len + max_new_tokens) * sizeof(int32_t));
    if (!seq_buf) {
      free(logits);
      return -1;
    }
    memcpy(seq_buf, prompt_ids, (size_t)prompt_len * sizeof(int32_t));
  }

  t_start = now_ns();
  if (max_new_tokens > 0) {
    if (use_kv_cache) {
      for (pos = 0; pos < prompt_len; pos++) {
        step_token(
            m, pos, prompt_ids[pos], binarize_tologic,
            (pos == prompt_len - 1) ? logits : NULL,
            step_prof);
      }
      t_after_prefill = now_ns();
      for (step = 0; step < max_new_tokens; step++) {
        int32_t next_tok = pick_next_token(logits, V);
        if (next_tok == EOS_ID) {
          eos_hit = 1;
          break;
        }
        if (out_tokens) out_tokens[count] = next_tok;
        count++;
        if (step + 1 < max_new_tokens) {
          step_token(
              m, prompt_len + step, next_tok, binarize_tologic,
              logits, step_prof);
        }
      }
    } else {
      t_after_prefill = t_start;
      for (step = 0; step < max_new_tokens; step++) {
        int32_t cur_len = prompt_len + step;
        int32_t next_tok;
        for (pos = 0; pos < cur_len; pos++) {
          step_token(
              m, pos, seq_buf[pos], binarize_tologic,
              (pos == cur_len - 1) ? logits : NULL,
              step_prof);
        }
        next_tok = pick_next_token(logits, V);
        if (next_tok == EOS_ID) {
          eos_hit = 1;
          break;
        }
        if (out_tokens) out_tokens[count] = next_tok;
        seq_buf[cur_len] = next_tok;
        count++;
      }
    }
  } else {
    t_after_prefill = t_start;
  }
  t_end = now_ns();

  *out_len = count;
  *out_eos = eos_hit;
  if (out_profile_ns7) {
    prof[0] = t_end - t_start;
    prof[1] = t_after_prefill - t_start;
    prof[2] = t_end - t_after_prefill;
    memcpy(out_profile_ns7, prof, sizeof(prof));
  }

  free(seq_buf);
  free(logits);
  return 0;
}
"""


def _shared_lib_filename() -> str:
    if sys.platform == "darwin":
        return "libmoduleio_runtime.dylib"
    if sys.platform.startswith("win"):
        return "moduleio_runtime.dll"
    return "libmoduleio_runtime.so"


def _extract_lut_indices(layer: torch.nn.Module) -> np.ndarray:
    if getattr(layer, "wiring", "fixed") == "learnable":
        idx = layer.learnable_mapping.weights.detach().cpu().argmax(dim=0)
        return idx.reshape(layer.out_dim, layer.n_inputs).numpy().astype("<u2")
    return layer.indices.detach().cpu().T.contiguous().numpy().astype("<u2")


def _extract_lut_truth_tables_u64(layer: torch.nn.Module) -> np.ndarray:
    raw = truth_tables_from_lut_layer(layer)
    return np.asarray([int(t) for t in raw], dtype="<u8")


def _compile_shared_lib(source_path: Path, lib_path: Path) -> list[str]:
    candidates = [
        ["cc", "-O3", "-mcpu=native", "-shared", "-fPIC", str(source_path), "-o", str(lib_path), "-lm"],
        ["cc", "-O3", "-shared", "-fPIC", str(source_path), "-o", str(lib_path), "-lm"],
    ]
    last_err = ""
    for cmd in candidates:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return cmd
        last_err = res.stderr.strip() or res.stdout.strip()
    raise RuntimeError(f"failed to compile C runtime: {last_err}")


def export_c_model(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    compile_shared: bool = True,
) -> dict[str, Any]:
    """Export a Griffith module-I/O checkpoint to native C99 + weights binary."""
    model, metadata = load_checkpoint(checkpoint_path, device="cpu")
    cfg = model.config
    if cfg.attention != "fp":
        raise ValueError(f"C export currently supports attention='fp', got {cfg.attention!r}")
    if cfg.ffn not in ("logic", "fp"):
        raise ValueError(f"unsupported ffn for C export: {cfg.ffn!r}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.bin"
    header_path = out_dir / "moduleio_runtime.h"
    source_path = out_dir / "moduleio_runtime.c"
    meta_path = out_dir / "export_meta.json"
    lib_path = out_dir / _shared_lib_filename()

    ffn_is_logic = int(cfg.ffn == "logic")
    header_words = [
        _MAGIC,
        _FORMAT_VERSION,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        cfg.mlp_ratio,
        VOCAB_SIZE,
        cfg.max_seq_len,
        ffn_is_logic,
        cfg.ffn_n_inputs if ffn_is_logic else 2,
        0, 0, 0, 0, 0, 0,
    ]
    chunks: list[bytes] = [struct.pack("<16i", *header_words)]

    def add_f32(t: torch.Tensor) -> None:
        chunks.append(np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4").tobytes())

    net = model.network
    add_f32(net.token_embed.weight)
    add_f32(net.pos_embed.weight)

    for block in net.blocks:
        add_f32(block.norm1.weight)
        add_f32(block.norm1.bias)
        add_f32(block.attn.attn.in_proj_weight)
        add_f32(block.attn.attn.in_proj_bias)
        add_f32(block.attn.attn.out_proj.weight)
        add_f32(block.attn.attn.out_proj.bias)
        add_f32(block.norm2.weight)
        add_f32(block.norm2.bias)

        if cfg.ffn == "logic":
            ffn = block.ffn
            if len(ffn.luts) != 2:
                raise ValueError(f"expected 2-layer LogicMLP FFN, got {len(ffn.luts)}")
            inv_tau = 1.0 / float(ffn.encode.log_tau.detach().cpu().exp().item())
            from_tau = float(ffn.decode.tau)
            chunks.append(struct.pack("<2f", inv_tau, from_tau))

            lut0 = ffn.luts[0]
            lut1 = ffn.luts[1]
            chunks.append(_extract_lut_indices(lut0).tobytes())
            add_f32(lut0._warp_weights())
            chunks.append(_extract_lut_truth_tables_u64(lut0).tobytes())
            chunks.append(_extract_lut_indices(lut1).tobytes())
            chunks.append(_extract_lut_truth_tables_u64(lut1).tobytes())
        else:
            add_f32(block.ffn[0].weight)
            add_f32(block.ffn[0].bias)
            add_f32(block.ffn[3].weight)
            add_f32(block.ffn[3].bias)

    add_f32(net.norm.weight)
    add_f32(net.norm.bias)
    add_f32(net.head.weight)
    add_f32(net.head.bias)

    weights_bytes = b"".join(chunks)
    weights_path.write_bytes(weights_bytes)
    header_path.write_text(_RUNTIME_H, encoding="utf-8")
    source_path.write_text(_RUNTIME_C, encoding="utf-8")

    compiler_cmd: list[str] | None = None
    if compile_shared:
        compiler_cmd = _compile_shared_lib(source_path, lib_path)

    export_meta = {
        "source_checkpoint": str(checkpoint_path),
        "model_config": {
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "mlp_ratio": cfg.mlp_ratio,
            "max_seq_len": cfg.max_seq_len,
            "attention": cfg.attention,
            "ffn": cfg.ffn,
            "ffn_n_inputs": cfg.ffn_n_inputs,
            "wiring": cfg.wiring,
            "wire_format": metadata.get("wire_format") or cfg.wire_format,
        },
        "weights_bytes": len(weights_bytes),
        "compiled": compile_shared,
        "compiler_cmd": compiler_cmd,
    }
    meta_path.write_text(json.dumps(export_meta, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(out_dir),
        "weights_path": str(weights_path),
        "weights_bytes": len(weights_bytes),
        "header_path": str(header_path),
        "source_path": str(source_path),
        "library_path": str(lib_path) if compile_shared else None,
        "compiled": compile_shared,
        "compiler_cmd": compiler_cmd,
        "model_config": export_meta["model_config"],
    }


class CompiledCModel:
    """Native C99 inference wrapper matching ModuleIOModel's greedy interface."""

    def __init__(self, export_dir: str | Path):
        out_dir = Path(export_dir)
        lib_path = out_dir / _shared_lib_filename()
        weights_path = out_dir / "weights.bin"
        if not lib_path.exists():
            raise FileNotFoundError(f"compiled shared library not found: {lib_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"weights binary not found: {weights_path}")

        self._lib = ctypes.CDLL(str(lib_path))
        self._configure_ctypes()
        self._handle = self._lib.moduleio_c_load(str(weights_path).encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"moduleio_c_load failed for {weights_path}")

        info = (ctypes.c_int32 * 8)()
        rc = self._lib.moduleio_c_info(self._handle, info)
        if rc != 0:
            self.close()
            raise RuntimeError("moduleio_c_info failed")
        self.d_model = int(info[0])
        self.n_heads = int(info[1])
        self.n_layers = int(info[2])
        self.mlp_ratio = int(info[3])
        self.vocab_size = int(info[4])
        self.max_seq_len = int(info[5])
        self.ffn_is_logic = bool(info[6])
        self.ffn_n_inputs = int(info[7])

        meta_path = out_dir / "export_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        mc = meta.get("model_config", {})
        self.config = ModelConfig(
            max_seq_len=self.max_seq_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_ratio=self.mlp_ratio,
            attention=mc.get("attention", "fp"),
            ffn=mc.get("ffn", "logic" if self.ffn_is_logic else "fp"),
            ffn_n_inputs=self.ffn_n_inputs,
            wiring=mc.get("wiring", "fixed"),
            wire_format=mc.get("wire_format", "compact-v2"),
        )

    def _configure_ctypes(self) -> None:
        lib = self._lib
        c_void_p = ctypes.c_void_p
        c_char_p = ctypes.c_char_p
        c_int = ctypes.c_int
        c_int32_p = ctypes.POINTER(ctypes.c_int32)
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_uint64_p = ctypes.POINTER(ctypes.c_uint64)

        lib.moduleio_c_load.argtypes = [c_char_p]
        lib.moduleio_c_load.restype = c_void_p
        lib.moduleio_c_free.argtypes = [c_void_p]
        lib.moduleio_c_free.restype = None
        lib.moduleio_c_info.argtypes = [c_void_p, c_int32_p]
        lib.moduleio_c_info.restype = c_int
        lib.moduleio_c_forward_last.argtypes = [
            c_void_p, c_int32_p, ctypes.c_int32, ctypes.c_int32, c_float_p,
        ]
        lib.moduleio_c_forward_last.restype = c_int
        lib.moduleio_c_generate.argtypes = [
            c_void_p, c_int32_p, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32,
            c_int32_p, c_int32_p, c_int32_p, c_uint64_p,
        ]
        lib.moduleio_c_generate.restype = c_int

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.moduleio_c_free(self._handle)
            self._handle = None

    def __enter__(self) -> CompiledCModel:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def forward_last_logits(
        self,
        token_ids: Sequence[int],
        *,
        binarize_tologic: bool = False,
    ) -> np.ndarray:
        seq = np.asarray(token_ids, dtype=np.int32)
        if seq.ndim != 1 or seq.size == 0:
            raise ValueError("token_ids must be a nonempty 1D sequence")
        if seq.size > self.max_seq_len:
            raise ExampleTooLongError("input exceeds max_seq_len; context is never cropped")
        out = np.empty(self.vocab_size, dtype=np.float32)
        rc = self._lib.moduleio_c_forward_last(
            self._handle,
            seq.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(seq.size),
            ctypes.c_int32(int(binarize_tologic)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError("moduleio_c_forward_last failed")
        return out

    def generate(
        self,
        input_text: str,
        max_new_tokens: int | None = None,
        *,
        use_kv_cache: bool = True,
        binarize_tologic: bool = False,
        profile: bool = False,
    ) -> dict[str, Any]:
        prefix = [BOS] + encode_text(input_text) + [SEP]
        capacity = self.max_seq_len
        remaining = capacity - len(prefix)
        from_context = max_new_tokens is None
        if max_new_tokens is None:
            max_new_tokens = remaining
        elif isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if remaining < 1:
            raise ExampleTooLongError(
                f"input uses {len(prefix)} tokens with max_seq_len={capacity}; "
                "no output token fits and the input will not be cropped"
            )
        if max_new_tokens > remaining:
            raise ExampleTooLongError(
                f"input {len(prefix)} + requested output budget {max_new_tokens} "
                f"exceeds max_seq_len={capacity}; maximum output budget is {remaining}"
            )

        prompt_arr = np.asarray(prefix, dtype=np.int32)
        out_arr = np.empty(max(1, max_new_tokens), dtype=np.int32)
        out_len = ctypes.c_int32(0)
        out_eos = ctypes.c_int32(0)
        prof_arr = (ctypes.c_uint64 * 7)() if profile else ctypes.POINTER(ctypes.c_uint64)()

        rc = self._lib.moduleio_c_generate(
            self._handle,
            prompt_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(len(prefix)),
            ctypes.c_int32(max_new_tokens),
            ctypes.c_int32(int(use_kv_cache)),
            ctypes.c_int32(int(binarize_tologic)),
            out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(out_len),
            ctypes.byref(out_eos),
            prof_arr,
        )
        if rc != 0:
            raise RuntimeError("moduleio_c_generate failed")

        generated = [int(v) for v in out_arr[: int(out_len.value)]]
        eos_generated = bool(out_eos.value)
        raw = bytes(value - BYTE_OFFSET for value in generated)
        try:
            text = raw.decode("utf-8", errors="strict")
            utf8_valid = True
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            utf8_valid = False
        result: dict[str, Any] = {
            "text": text,
            "token_ids": generated,
            "stop_reason": ("eos" if eos_generated else
                            "context_limit" if from_context else "max_new_tokens"),
            "eos_generated": eos_generated,
            "utf8_valid": utf8_valid,
            "input_tokens": len(prefix),
            "generated_tokens": len(generated) + int(eos_generated),
            "output_bytes": len(raw),
            "max_new_tokens": max_new_tokens,
            "max_seq_len": capacity,
            "mode": "c_hard",
        }
        if profile:
            result["profile_ns"] = {
                "total_ns": int(prof_arr[0]),
                "prefill_ns": int(prof_arr[1]),
                "decode_ns": int(prof_arr[2]),
                "attn_ns": int(prof_arr[3]),
                "lut0_or_ffn_ns": int(prof_arr[4]),
                "lut1_and_fromlogic_ns": int(prof_arr[5]),
                "head_ns": int(prof_arr[6]),
            }
        return result
