/*
 * Copyright (c) 2025
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * DNN ggml backend implementation.
 */
#include "libavutil/opt.h"
#include <ggml.h>
#include "dnn_backend_common.h"

static const AVOption dnn_ggml_options[] = {
    { NULL }
};

static DNNModel *dnn_load_model_gg(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    ggml_time_init();
    return NULL;
}

static int dnn_execute_model_gg(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    return 0;
}

static DNNAsyncStatusType dnn_get_result_gg(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    return DAST_FAIL;
}

static int dnn_flush_gg(const DNNModel *model)
{
    return 0;
}

static void dnn_free_model_gg(DNNModel **model)
{
}

const DNNModule ff_dnn_backend_ggml = {
    .clazz          = DNN_DEFINE_CLASS(dnn_ggml),
    .type           = DNN_GG,
    .load_model     = dnn_load_model_gg,
    .execute_model  = dnn_execute_model_gg,
    .get_result     = dnn_get_result_gg,
    .flush          = dnn_flush_gg,
    .free_model     = dnn_free_model_gg,
};
