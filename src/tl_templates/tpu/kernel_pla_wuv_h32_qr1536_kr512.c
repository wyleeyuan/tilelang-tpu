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
  __ppl_tensor_info v8 = {.shape = {1, 32, 1, 128} , .stride = {0}, .addr = v4, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 8192, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v7 = {.shape = {1, 8192, 1, 4} , .stride = {0}, .addr = v3, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 65536, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v6 = {.shape = {1, 8192, 1, 512} , .stride = {0}, .addr = v2, .dtype = DT_FP8E4M3, .mode = 2, .align_mode = 0, .size = 4194304, .unsigned_flag = 0, .default_stride = true};
  __ppl_tensor_info v5 = {.shape = {1, 32, 1, 512} , .stride = {0}, .addr = v1, .dtype = DT_BFP16, .mode = 2, .align_mode = 0, .size = 32768, .unsigned_flag = 0, .default_stride = true};
  for (int h = 0; h < 32; ++h) {
    __ppl_tensor_info acc = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 0, .dtype = DT_FP32, .mode = 2, .align_mode = 1, .size = 512, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&acc.stride, 0, &acc.shape, DT_FP32);
    __ppl_tensor_info out_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16384, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
    tpu_aligned_stride(&out_shared.stride, 0, &out_shared.shape, DT_BFP16);
    {
    scalar_t acc_scalar_f32 = {.f32 = 0.000000e+00f};
    tpu_bdc_set_C(acc.addr, acc_scalar_f32, &acc.shape, (acc.default_stride ? NULL : &acc.stride), DT_FP32);
    }
    for (int ib = 0; ib < 4; ++ib) {
      __ppl_tensor_info latent_shared = {.shape = { 1, 1, 1, 128}, .stride = {0}, .addr = 16896, .dtype = DT_BFP16, .mode = 2, .align_mode = 1, .size = 256, .unsigned_flag = 0, .default_stride = false};
      tpu_aligned_stride(&latent_shared.stride, 0, &latent_shared.shape, DT_BFP16);
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
      __ppl_tensor_info latent = {.shape = {1, 1, 1, 128} , .stride = {16384, 512, 512, 1} , .addr = v5.addr + ((0) * 16384+(h) * 512+(0) * 512+((ib * 128)) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 16384+(h) * 512+(0) * 512+((ib * 128)) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info latent_shared_1 = {.shape = {1, 1, 1, 128} , .stride = latent_shared.stride, .addr = latent_shared.addr + ((0) * latent_shared.stride.c+(0) * latent_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * latent_shared.stride.c+(0) * latent_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = latent_shared.default_stride};
      tpu_gdma_cpy_S2L(latent_shared_1.addr, latent.addr, &latent_shared_1.shape, (latent_shared_1.default_stride ? NULL : &latent_shared_1.stride), (latent.default_stride ? NULL : &latent.stride), DT_BFP16);
      __ppl_tensor_info wukv = {.shape = {1, 128, 1, 128} , .stride = {4194304, 512, 512, 1} , .addr = v6.addr + ((((h * 256) + 128)) * 512+((ib * 128)) * 1 ) * 1, .dtype = DT_FP8E4M3, .mode = 2, .size = 1, .offset = ((((h * 256) + 128)) * 512+((ib * 128)) * 1 ) * 1, .unsigned_flag = 0, .default_stride = false};
      __ppl_tensor_info weight_fp8_1 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8.stride, .addr = weight_fp8.addr + ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8.default_stride};
      tpu_gdma_cpy_S2L(weight_fp8_1.addr, wukv.addr, &weight_fp8_1.shape, (weight_fp8_1.default_stride ? NULL : &weight_fp8_1.stride), (wukv.default_stride ? NULL : &wukv.stride), DT_FP8E4M3);
      __ppl_tensor_info weight_fp8_2 = {.shape = {1, 128, 1, 128} , .stride = weight_fp8.stride, .addr = weight_fp8.addr + ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .dtype = DT_FP8E4M3, .mode = 0, .size = 1, .offset = ((0) * weight_fp8.stride.c+(0) * weight_fp8.stride.w ) * 1, .unsigned_flag = 0, .default_stride = weight_fp8.default_stride};
      __ppl_tensor_info weight_bf16_1 = {.shape = {1, 128, 1, 128} , .stride = weight_bf16.stride, .addr = weight_bf16.addr + ((0) * weight_bf16.stride.c+(0) * weight_bf16.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * weight_bf16.stride.c+(0) * weight_bf16.stride.w ) * 2, .unsigned_flag = 0, .default_stride = weight_bf16.default_stride};
      tpu_bdc_cast(weight_bf16_1.addr, weight_fp8_2.addr, &weight_bf16_1.shape, (weight_bf16_1.default_stride ? NULL : &weight_bf16_1.stride), (weight_fp8_2.default_stride ? NULL : &weight_fp8_2.stride), DT_BFP16, DT_FP8E4M3, RM_HALF_TO_EVEN);
      __ppl_tensor_info expanded_scale = {.shape = {1, 128, 1, 1} , .stride = {32768, 4, 4, 1} , .addr = v7.addr + ((((h * 256) + 128)) * 4+(ib) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((((h * 256) + 128)) * 4+(ib) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
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
      tpu_bdc_fp_mm_R_trans(partial.addr, latent_shared.addr, weight_dequant.addr, 1, 128, 128, DT_FP32, DT_BFP16);
      tpu_bdc_fp_add( acc.addr, acc.addr, partial.addr, &acc.shape, (acc.default_stride ? NULL : &acc.stride), (acc.default_stride ? NULL : &acc.stride), (partial.default_stride ? NULL : &partial.stride), DT_FP32);
    }
    __ppl_tensor_info acc_1 = {.shape = {1, 1, 1, 128} , .stride = acc.stride, .addr = acc.addr + ((0) * acc.stride.c+(0) * acc.stride.w ) * 4, .dtype = DT_FP32, .mode = 0, .size = 1, .offset = ((0) * acc.stride.c+(0) * acc.stride.w ) * 4, .unsigned_flag = 0, .default_stride = acc.default_stride};
    __ppl_tensor_info out_shared_1 = {.shape = {1, 1, 1, 128} , .stride = out_shared.stride, .addr = out_shared.addr + ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_shared.default_stride};
    tpu_bdc_cast(out_shared_1.addr, acc_1.addr, &out_shared_1.shape, (out_shared_1.default_stride ? NULL : &out_shared_1.stride), (acc_1.default_stride ? NULL : &acc_1.stride), DT_BFP16, DT_FP32, RM_HALF_TO_EVEN);
    __ppl_tensor_info out_shared_2 = {.shape = {1, 1, 1, 128} , .stride = out_shared.stride, .addr = out_shared.addr + ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .dtype = DT_BFP16, .mode = 0, .size = 1, .offset = ((0) * out_shared.stride.c+(0) * out_shared.stride.w ) * 2, .unsigned_flag = 0, .default_stride = out_shared.default_stride};
    __ppl_tensor_info output = {.shape = {1, 1, 1, 128} , .stride = {4096, 128, 128, 1} , .addr = v8.addr + ((0) * 4096+(h) * 128+(0) * 128+(0) * 1 ) * 2, .dtype = DT_BFP16, .mode = 2, .size = 1, .offset = ((0) * 4096+(h) * 128+(0) * 128+(0) * 1 ) * 2, .unsigned_flag = 0, .default_stride = false};
    tpu_gdma_cpy_L2S(output.addr, out_shared_2.addr, &output.shape, (output.default_stride ? NULL : &output.stride), (out_shared_2.default_stride ? NULL : &out_shared_2.stride), DT_BFP16);
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
