#include "ppl_helper.h"
static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,    DT_FP32,    DT_FP16,  DT_BFP16,
    DT_FP8E5M2, DT_FP8E4M3, DT_FP20,  DT_TF32,
    DT_INT32,   DT_UINT32,  DT_INT16, DT_UINT16,
    DT_INT8,    DT_UINT8,   DT_INT4,  DT_UINT4};
  return __dtype[type];
}

typedef struct {
    dim4 shape;
    dim4 stride;
    global_addr_t addr;
    data_type_t dtype;
    int mode;
    int align_mode;
    int size;
    int offset;
    bool unsigned_flag;
    bool default_stride;
} __ppl_tensor_info;

void main_kernel_inner(global_addr_t v1, global_addr_t v2, global_addr_t v3, global_addr_t v4) {
  __ppl_tensor_info v8 = {.shape = {1, 1, 1, 6144} , .stride = {0}, .addr = v4, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 12288, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {1, 6144, 1, 12} , .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 147456, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {1, 6144, 1, 1536} , .stride = {0}, .addr = v2, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 9437184, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1, 1, 1, 1536} , .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 3072, .unsigned_flag = 0, .default_stride = true};
  for (int ob = 0; ob < 48; ++ob) {
    __ppl_tensor_info acc = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&acc.stride, 0, &acc.shape, DT_FP32);
    __ppl_tensor_info out_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_shared.stride, 0, &out_shared.shape, DT_BFP16);
    {
    scalar_t acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(acc.addr, acc_scalar_f32, &acc.shape, (acc.default_stride ? NULL : &acc.stride), DT_FP32);
    }
    for (int ib = 0; ib < 12; ++ib) {
      __ppl_tensor_info query_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16896, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&query_shared.stride, 0, &query_shared.shape, DT_BFP16);
      __ppl_tensor_info weight_fp8 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_fp8.stride, 0, &weight_fp8.shape, DT_FP8E4M3);
      __ppl_tensor_info weight_bf16 = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 512, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_bf16.stride, 0, &weight_bf16.shape, DT_BFP16);
      __ppl_tensor_info scale_rows = {.shape = { 1, 128, 1, 1}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 128, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&scale_rows.stride, 0, &scale_rows.shape, DT_BFP16);
      __ppl_tensor_info weight_dequant = {.shape = { 1, 128, 1, 128}, .stride = {0}, .addr = 1024, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&weight_dequant.stride, 0, &weight_dequant.shape, DT_BFP16);
      __ppl_tensor_info partial = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&partial.stride, 0, &partial.shape, DT_FP32);
      __ppl_tensor_info query = {.shape = {1, 1, 1, 128} , .stride = {1536, 1536, 1536, 1} , .addr = v5.addr + ((0) * 1536+((ib * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 1536+((ib * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info query_shared_1 = {.shape = {1, 1, 1, 128} , .stride = query_shared.stride, .addr = query_shared.addr + ((0) * query_shared.stride.c+(0) * query_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * query_shared.stride.c+(0) * query_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = query_shared.default_stride};
      tpu_gdma_cpy_S2L(query_shared_1.addr, query.addr, &query_shared_1.shape, (query_shared_1.default_stride ? NULL : &query_shared_1.stride), (query.default_stride ? NULL : &query.stride), DT_BFP16);
      __ppl_tensor_info wuq = {.shape = {1, 128, 1, 128} , .stride = {9437184, 1536, 1536, 1} , .addr = v6.addr + (((ob * 128)) * 1536+((ib * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = (((ob * 128)) * 1536+((ib * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info weight_fp8_1 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8.stride, .addr = weight_fp8.addr + ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8.default_stride};
      tpu_gdma_cpy_S2L(weight_fp8_1.addr, wuq.addr, &weight_fp8_1.shape, (weight_fp8_1.default_stride ? NULL : &weight_fp8_1.stride), (wuq.default_stride ? NULL : &wuq.stride), DT_FP8E4M3);
      __ppl_tensor_info weight_fp8_2 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8.stride, .addr = weight_fp8.addr + ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8.default_stride};
      __ppl_tensor_info weight_bf16_1 = {.shape = {1, 128, 1, 128} , .stride = weight_bf16.stride, .addr = weight_bf16.addr + ((0) * weight_bf16.stride.c+(0) * weight_bf16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * weight_bf16.stride.c+(0) * weight_bf16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = weight_bf16.default_stride};
      tpu_bdc_cast(weight_bf16_1.addr, weight_fp8_2.addr, &weight_bf16_1.shape, (weight_bf16_1.default_stride ? NULL : &weight_bf16_1.stride), (weight_fp8_2.default_stride ? NULL : &weight_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info expanded_scale = {.shape = {1, 128, 1, 1} , .stride = {73728, 12, 12, 1} , .addr = v7.addr + (((ob * 128)) * 12+(ib) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = (((ob * 128)) * 12+(ib) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info scale_rows_1 = {.shape = {1, 128, 1, 1} , .stride = scale_rows.stride, .addr = scale_rows.addr + ((0) * scale_rows.stride.c+(0) * scale_rows.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * scale_rows.stride.c+(0) * scale_rows.stride.w ) * 2, .unsigned_flag = 0, .default_stride = scale_rows.default_stride};
      tpu_gdma_cpy_S2L(scale_rows_1.addr, expanded_scale.addr, &scale_rows_1.shape, (scale_rows_1.default_stride ? NULL : &scale_rows_1.stride), (expanded_scale.default_stride ? NULL : &expanded_scale.stride), DT_BFP16);
      dim4 scale_rows_stride;
      tpu_aligned_stride(&scale_rows_stride, 0, &scale_rows.shape, DT_BFP16);
      scale_rows_stride.w = 0;
      tpu_bdc_fp_mul( weight_dequant.addr, weight_bf16.addr, scale_rows.addr, &weight_dequant.shape, (weight_dequant.default_stride ? NULL : &weight_dequant.stride), (weight_bf16.default_stride ? NULL : &weight_bf16.stride), &scale_rows_stride, DT_BFP16);
      {
      scalar_t partial_scalar_f32 = {.f32 = 0.000000e+00f};
      tpu_bdc_set_C(partial.addr, partial_scalar_f32, &partial.shape, (partial.default_stride ? NULL : &partial.stride), DT_FP32);
      }
      tpu_bdc_fp_mm_R_trans(partial.addr, query_shared.addr, weight_dequant.addr, 1, 128, 128, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( acc.addr, acc.addr, partial.addr, &acc.shape, (acc.default_stride ? NULL : &acc.stride), (acc.default_stride ? NULL : &acc.stride), (partial.default_stride ? NULL : &partial.stride), DT_FP32);
    }
    __ppl_tensor_info acc_1 = {.shape = {1, 1, 1, 128} , .stride = acc.stride, .addr = acc.addr + ((0) * acc.stride.c+(0) * acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * acc.stride.c+(0) * acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = acc.default_stride};
    __ppl_tensor_info out_shared_1 = {.shape = {1, 1, 1, 128} , .stride = out_shared.stride, .addr = out_shared.addr + ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_shared.default_stride};
    tpu_bdc_cast(out_shared_1.addr, acc_1.addr, &out_shared_1.shape, (out_shared_1.default_stride ? NULL : &out_shared_1.stride), (acc_1.default_stride ? NULL : &acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_shared_2 = {.shape = {1, 1, 1, 128} , .stride = out_shared.stride, .addr = out_shared.addr + ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_shared.default_stride};
    __ppl_tensor_info q_full = {.shape = {1, 1, 1, 128} , .stride = {6144, 6144, 6144, 1} , .addr = v8.addr + ((0) * 6144+((ob * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 6144+((ob * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(q_full.addr, out_shared_2.addr, &q_full.shape, (q_full.default_stride ? NULL : &q_full.stride), (out_shared_2.default_stride ? NULL : &out_shared_2.stride), DT_BFP16);
  }
}

typedef struct {
  global_addr_t v1;
  global_addr_t v2;
  global_addr_t v3;
  global_addr_t v4;
} tpu_kernel_api_main_inner_args_t;
int main_kernel(const void * args) {
  tpu_kernel_api_main_inner_args_t *api = (tpu_kernel_api_main_inner_args_t*)args;
  tpu_initialize();
  main_kernel_inner(api->v1,
    api->v2,
    api->v3,
    api->v4);
  tpu_poll();
  return 0;
}
TPUKERNEL_FUNC_REGISTER(main_kernel)
