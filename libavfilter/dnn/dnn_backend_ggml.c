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
#include "libavutil/avassert.h"
#include "libavutil/cpu.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "safe_queue.h"
#include <ggml/ggml-dnn.h>

typedef struct GGModel {
    DNNModel model;
    DnnContext *ctx;
    ggml_dnn_model_t gg_model;
    //TF_Graph *graph;
    //TF_Session *session;
    //TF_Status *status;
    SafeQueue *request_queue;
    Queue *lltask_queue;
    Queue *task_queue;
} GGModel;

/**
 * Stores execution parameters for single
 * call to the ggml C API
 */
typedef struct GGInferRequest {
    //TF_Output *tf_outputs;
    //TF_Tensor **output_tensors;
    //TF_Output *tf_input;
    //TF_Tensor *input_tensor;
    ggml_dnn_tensor_t input_tensor;
    int nb_output;
    ggml_dnn_tensor_t *output_tensors;
} GGInferRequest;

typedef struct GGRequestItem {
    GGInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    //TF_Status *status;
    DNNAsyncExecModule exec_module;
} GGRequestItem;

#define OFFSET(x) offsetof(GGOptions, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_ggml_options[] = {
    { "backend_dir", "path to ggml backend dir", OFFSET(backend_dir), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, FLAGS },
    { NULL }
};

static int execute_model_gg(GGRequestItem *request, Queue *lltask_queue);
static void infer_completion_callback(void *args);
//static inline void destroy_request_item(GGRequestItem **arg);

static void gg_free_request(GGInferRequest *request)
{
    if (!request)
        return;
    if (request->input_tensor) {
        ggml_dnn_free_tensor(request->input_tensor);
        //TF_DeleteTensor(request->input_tensor);
        request->input_tensor = NULL;
    }
    // av_freep(&request->tf_input);
    // av_freep(&request->tf_outputs);
    if (request->output_tensors) {
        for (uint32_t i = 0; i < request->nb_output; ++i) {
            if (request->output_tensors[i]) {
                ggml_dnn_free_tensor(request->output_tensors[i]);
                //TF_DeleteTensor(request->output_tensors[i]);
                request->output_tensors[i] = NULL;
            }
        }
        av_freep(&request->output_tensors);
    }
}

static GGInferRequest *gg_create_inference_request(void)
{
    GGInferRequest *infer_request = av_malloc(sizeof(GGInferRequest));
    if (!infer_request) {
        return NULL;
    }
    // infer_request->tf_outputs = NULL;
    // infer_request->tf_input = NULL;
    infer_request->input_tensor = NULL;
    infer_request->output_tensors = NULL;
    return infer_request;
}

static int gg_start_inference(void *args)
{
    GGRequestItem *request = args;
    GGInferRequest *infer_request = request->infer_request;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    GGModel *gg_model = task->model;

    if (!request) {
        av_log(gg_model->ctx, AV_LOG_ERROR, "GGRequestItem is NULL\n");
        return AVERROR(EINVAL);
    }
    printf(">>> running inference for GGModel nb_output: %d\n", infer_request->nb_output);
    int ret = ggml_dnn_infer_model(gg_model->gg_model, 
                                   1, &infer_request->input_tensor,
                                   infer_request->nb_output, infer_request->output_tensors);
    if (ret != GGML_DNN_OK) {
        av_log(gg_model->ctx, AV_LOG_ERROR, "Failed to run inference: %d\n", ret);
        return DNN_GENERIC_ERROR;
    }
    // TF_SessionRun(tf_model->session, NULL,
    //               infer_request->tf_input, &infer_request->input_tensor, 1,
    //               infer_request->tf_outputs, infer_request->output_tensors,
    //               task->nb_output, NULL, 0, NULL,
    //               request->status);
    // if (TF_GetCode(request->status) != TF_OK) {
    //     av_log(tf_model->ctx, AV_LOG_ERROR, "%s", TF_Message(request->status));
    //     return DNN_GENERIC_ERROR;
    // }
    return 0;
}

static inline void destroy_request_item(GGRequestItem **arg) {
    GGRequestItem *request;
    if (!arg) {
        return;
    }
    request = *arg;
    gg_free_request(request->infer_request);
    av_freep(&request->infer_request);
    av_freep(&request->lltask);
    //TF_DeleteStatus(request->status);
    ff_dnn_async_module_cleanup(&request->exec_module);
    av_freep(arg);
}

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue)
{
    GGModel *gg_model = task->model;
    DnnContext *ctx = gg_model->ctx;
    LastLevelTaskItem *lltask = av_malloc(sizeof(*lltask));
    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR, "Unable to allocate space for LastLevelTaskItem\n");
        return AVERROR(ENOMEM);
    }
    task->inference_todo = 1;
    task->inference_done = 0;
    lltask->task = task;
    if (ff_queue_push_back(lltask_queue, lltask) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
        av_freep(&lltask);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static int get_input_gg(DNNModel *model, DNNData *input, const char *input_name)
{
    GGModel *gg_model = (GGModel *)model;
    DnnContext *ctx = gg_model->ctx;
    struct ggml_dnn_tensor_desc desc;

    printf(">>>> get_input_gg: %s\n", input_name);
    // TF_Status *status;
    // TF_DataType dt;
    //int64_t dims[4];

    // TODO: use input_name instead of 0
    int ret = ggml_dnn_model_input_desc(gg_model->gg_model, 0, &desc);
    if (ret != GGML_DNN_OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get input tensor description: %d\n", ret);
        return DNN_GENERIC_ERROR;
    }

    // TF_Output tf_output;
    // tf_output.oper = TF_GraphOperationByName(tf_model->graph, input_name);
    // if (!tf_output.oper) {
    //     av_log(ctx, AV_LOG_ERROR, "Could not find \"%s\" in model\n", input_name);
    //     return AVERROR(EINVAL);
    // }

    // tf_output.index = 0;
    // dt = TF_OperationOutputType(tf_output);
    switch (desc.type) {
    case F32:
        input->dt = DNN_FLOAT;
        break;
    case U8:
        input->dt = DNN_UINT8;
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported output type %d in model\n", desc.type);
        return AVERROR(EINVAL);
    }
    input->order = DCO_RGB;

    // status = TF_NewStatus();
    // TF_GraphGetTensorShape(tf_model->graph, tf_output, dims, 4, status);
    // if (TF_GetCode(status) != TF_OK){
    //     TF_DeleteStatus(status);
    //     av_log(ctx, AV_LOG_ERROR, "Failed to get input tensor shape: number of dimension incorrect\n");
    //     return DNN_GENERIC_ERROR;
    // }
    // TF_DeleteStatus(status);

    // currently only NHWC is supported
    av_assert0(desc.dims[0] == 1);
    for (int i = 0; i < 4; i++)
        input->dims[i] = desc.dims[i];
    input->layout = DL_NHWC;
    return 0;
}

static int get_output_gg(DNNModel *model, const char *input_name, int input_width, int input_height,
                                   const char *output_name, int *output_width, int *output_height)
{
    int ret;
    GGModel *gg_model = (GGModel *)model;
    DnnContext *ctx = gg_model->ctx;
    TaskItem task;
    GGRequestItem *request;
    DNNExecBaseParams exec_params = {
        .input_name     = input_name,
        .output_names   = &output_name,
        .nb_output      = 1,
        .in_frame       = NULL,
        .out_frame      = NULL,
    };

    ret = ff_dnn_fill_gettingoutput_task(&task, &exec_params, gg_model, input_height, input_width, ctx);
    printf(">> task->nb_output: %d\n", task.nb_output);
    if (ret != 0) {
        goto err;
    }

    ret = extract_lltask_from_task(&task, gg_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract inference from task.\n");
        goto err;
    }

    request = ff_safe_queue_pop_front(gg_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        ret = AVERROR(EINVAL);
        goto err;
    }

    ret = execute_model_gg(request, gg_model->lltask_queue);
    *output_width = task.out_frame->width;
    *output_height = task.out_frame->height;

err:
    av_frame_free(&task.out_frame);
    av_frame_free(&task.in_frame);
    return ret;
}

static int load_gg_model(GGModel *gg_model, const char *model_filename)
{
    ggml_dnn_init(gg_model->ctx->ggml_option.backend_dir, NULL);
    int ret = ggml_dnn_load_model(model_filename, &gg_model->gg_model);
    if (ret != GGML_DNN_OK) {
        return DNN_GENERIC_ERROR;
    }
    return 0;
}

static void dnn_free_model_gg(DNNModel **model)
{
    GGModel *gg_model;

    if (!model || !*model)
        return;

    gg_model = (GGModel *)(*model);
    while (ff_safe_queue_size(gg_model->request_queue) != 0) {
        GGRequestItem *item = ff_safe_queue_pop_front(gg_model->request_queue);
        destroy_request_item(&item);
    }
    ff_safe_queue_destroy(gg_model->request_queue);

    while (ff_queue_size(gg_model->lltask_queue) != 0) {
        LastLevelTaskItem *item = ff_queue_pop_front(gg_model->lltask_queue);
        av_freep(&item);
    }
    ff_queue_destroy(gg_model->lltask_queue);

    while (ff_queue_size(gg_model->task_queue) != 0) {
        TaskItem *item = ff_queue_pop_front(gg_model->task_queue);
        av_frame_free(&item->in_frame);
        av_frame_free(&item->out_frame);
        av_freep(&item);
    }
    ff_queue_destroy(gg_model->task_queue);

    if (gg_model->gg_model) {
        ggml_dnn_free_model(gg_model->gg_model);
    }
    // if (tf_model->graph){
    //     TF_DeleteGraph(gg_model->graph);
    // }
    // if (tf_model->session){
    //     TF_CloseSession(tf_model->session, tf_model->status);
    //     TF_DeleteSession(tf_model->session, tf_model->status);
    // }
    // if (tf_model->status){
    //     TF_DeleteStatus(tf_model->status);
    // }
    av_freep(&gg_model);
    *model = NULL;
}

static DNNModel *dnn_load_model_gg(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    DNNModel *model = NULL;
    GGModel *gg_model = NULL;

    gg_model = av_mallocz(sizeof(GGModel));
    if (!gg_model)
        return NULL;
    model = &gg_model->model;
    gg_model->ctx = ctx;

    if (load_gg_model(gg_model, ctx->model_filename) != 0){
        av_log(ctx, AV_LOG_ERROR, "Failed to load ggml model: \"%s\"\n", ctx->model_filename);
        goto err;
    }
    printf(">> Loaded ggml model: %s\n", ctx->model_filename);

    if (ctx->nireq <= 0) {
        ctx->nireq = av_cpu_count() / 2 + 1;
    }

#if !HAVE_PTHREAD_CANCEL
    if (ctx->options.async) {
        ctx->options.async = 0;
        av_log(filter_ctx, AV_LOG_WARNING, "pthread is not supported, roll back to sync.\n");
    }
#endif

    gg_model->request_queue = ff_safe_queue_create();
    if (!gg_model->request_queue) {
        goto err;
    }

    for (int i = 0; i < ctx->nireq; i++) {
        GGRequestItem *item = av_mallocz(sizeof(*item));
        if (!item) {
            goto err;
        }
        item->lltask = NULL;
        item->infer_request = gg_create_inference_request();
        if (!item->infer_request) {
            av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for ggml inference request\n");
            av_freep(&item);
            goto err;
        }
        // item->status = TF_NewStatus(); TODO: FIXME
        item->exec_module.start_inference = &gg_start_inference;
        item->exec_module.callback = &infer_completion_callback;
        item->exec_module.args = item;

        if (ff_safe_queue_push_back(gg_model->request_queue, item) < 0) {
            destroy_request_item(&item);
            goto err;
        }
    }

    gg_model->lltask_queue = ff_queue_create();
    if (!gg_model->lltask_queue) {
        goto err;
    }

    gg_model->task_queue = ff_queue_create();
    if (!gg_model->task_queue) {
        goto err;
    }

    model->get_input = &get_input_gg;
    model->get_output = &get_output_gg;
    model->filter_ctx = filter_ctx;
    model->func_type = func_type;

    return model;
err:
    dnn_free_model_gg(&model);
    return NULL;
}

static int fill_model_input_gg(GGModel *gg_model, GGRequestItem *request) {
    DNNData input = { 0 };
    LastLevelTaskItem *lltask;
    TaskItem *task;
    GGInferRequest *infer_request = NULL;
    DnnContext *ctx = gg_model->ctx;
    int ret = 0;

    lltask = ff_queue_pop_front(gg_model->lltask_queue);
    av_assert0(lltask);
    task = lltask->task;
    request->lltask = lltask;

    ret = get_input_gg(&gg_model->model, &input, task->input_name);
    if (ret != 0) {
        goto err;
    }

    infer_request = request->infer_request;
    input.dims[1] = task->in_frame->height;
    input.dims[2] = task->in_frame->width;

    // infer_request->tf_input = av_malloc(sizeof(TF_Output));
    // if (!infer_request->tf_input) {
    //     av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for input tensor\n");
    //     ret = AVERROR(ENOMEM);
    //     goto err;
    // }

    // infer_request->tf_input->oper = TF_GraphOperationByName(tf_model->graph, task->input_name);
    // if (!infer_request->tf_input->oper){
    //     av_log(ctx, AV_LOG_ERROR, "Could not find \"%s\" in model\n", task->input_name);
    //     ret = DNN_GENERIC_ERROR;
    //     goto err;
    // }
    // infer_request->tf_input->index = 0;

    // infer_request->input_tensor = allocate_input_tensor(&input);
    // if (!infer_request->input_tensor){
    //     av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for input tensor\n");
    //     ret = AVERROR(ENOMEM);
    //     goto err;
    // }
    // input.data = (float *)TF_TensorData(infer_request->input_tensor);

    struct ggml_dnn_tensor_desc input_desc;
    ret = ggml_dnn_model_input_desc(gg_model->gg_model, 0, &input_desc);
    if (ret != GGML_DNN_OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get input tensor description\n");
        return DNN_GENERIC_ERROR;
    }
    infer_request->input_tensor = ggml_dnn_alloc_tensor(input_desc);
    if (!infer_request->input_tensor) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for input tensor\n");
        ret = AVERROR(ENOMEM);
        goto err;
    }
    input.data = ggml_dnn_tensor_data(infer_request->input_tensor);

    switch (gg_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            if (gg_model->model.frame_pre_proc != NULL) {
                gg_model->model.frame_pre_proc(task->in_frame, &input, gg_model->model.filter_ctx);
            } else {
                ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
            }
        }
        break;
    case DFT_ANALYTICS_DETECT:
        ff_frame_to_dnn_detect(task->in_frame, &input, ctx);
        break;
    default:
        avpriv_report_missing_feature(ctx, "model function type %d", gg_model->model.func_type);
        break;
    }

    // infer_request->tf_outputs = av_malloc_array(task->nb_output, sizeof(TF_Output));
    // if (infer_request->tf_outputs == NULL) {
    //     av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for *tf_outputs\n");
    //     ret = AVERROR(ENOMEM);
    //     goto err;
    // }

    // infer_request->output_tensors = av_calloc(task->nb_output, sizeof(*infer_request->output_tensors));
    // if (!infer_request->output_tensors) {
    //     av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for output tensor\n");
    //     ret = AVERROR(ENOMEM);
    //     goto err;
    // }

    infer_request->nb_output = 2; // FIXME
    infer_request->output_tensors = av_calloc(infer_request->nb_output, sizeof(ggml_dnn_tensor_t));

    for (int i = 0; i < infer_request->nb_output; ++i) {
        struct ggml_dnn_tensor_desc output_desc;
        ret = ggml_dnn_model_output_desc(gg_model->gg_model, i, &output_desc);
        if (ret != GGML_DNN_OK) {
            av_log(ctx, AV_LOG_ERROR, "Failed to get output tensor description: %d\n", ret);
            return DNN_GENERIC_ERROR;
        }
        infer_request->output_tensors[i] = ggml_dnn_alloc_tensor(output_desc);
        if (!infer_request->output_tensors[i]) {
            av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for output tensor\n");
            ret = AVERROR(ENOMEM);
            goto err;
        }
        // infer_request->output_tensors[i] = NULL;
        //infer_request->tf_outputs[i].oper = TF_GraphOperationByName(tf_model->graph, task->output_names[i]);
        // if (!infer_request->tf_outputs[i].oper) {
        //     av_log(ctx, AV_LOG_ERROR, "Could not find output \"%s\" in model\n", task->output_names[i]);
        //     ret = DNN_GENERIC_ERROR;
        //     goto err;
        // }
        // infer_request->tf_outputs[i].index = 0;
    }

    return 0;
err:
    gg_free_request(infer_request);
    return ret;
}

static void infer_completion_callback(void *args) {
    GGRequestItem *request = args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    DNNData *outputs;
    GGInferRequest *infer_request = request->infer_request;
    GGModel *gg_model = task->model;
    DnnContext *ctx = gg_model->ctx;

    printf(">> infer_completion_callback, task->nb_output: %d\n", task->nb_output);
    outputs = av_calloc(task->nb_output, sizeof(*outputs));
    if (!outputs) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for *outputs\n");
        goto err;
    }

    for (uint32_t i = 0; i < task->nb_output; ++i) {
        // outputs[i].dims[dnn_get_height_idx_by_layout(outputs[i].layout)] =
        //     TF_Dim(infer_request->output_tensors[i], 1);
        // outputs[i].dims[dnn_get_width_idx_by_layout(outputs[i].layout)] =
        //     TF_Dim(infer_request->output_tensors[i], 2);
        // outputs[i].dims[dnn_get_channel_idx_by_layout(outputs[i].layout)] =
        //     TF_Dim(infer_request->output_tensors[i], 3);
        // outputs[i].data = TF_TensorData(infer_request->output_tensors[i]);
        // outputs[i].dt = (DNNDataType)TF_TensorType(infer_request->output_tensors[i]);
    }
    switch (gg_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        //it only support 1 output if it's frame in & frame out
        if (task->do_ioproc) {
            if (gg_model->model.frame_post_proc != NULL) {
                gg_model->model.frame_post_proc(task->out_frame, outputs, gg_model->model.filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, outputs, ctx);
            }
        } else {
            task->out_frame->width =
                outputs[0].dims[dnn_get_width_idx_by_layout(outputs[0].layout)];
            task->out_frame->height =
                outputs[0].dims[dnn_get_height_idx_by_layout(outputs[0].layout)];
        }
        break;
    case DFT_ANALYTICS_DETECT:
        if (!gg_model->model.detect_post_proc) {
            av_log(ctx, AV_LOG_ERROR, "Detect filter needs provide post proc\n");
            return;
        }
        gg_model->model.detect_post_proc(task->in_frame, outputs, 2, gg_model->model.filter_ctx); //FIXME
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "ggml backend does not support this kind of dnn filter now\n");
        goto err;
    }
    task->inference_done++;
err:
    gg_free_request(infer_request);
    av_freep(&outputs);

    if (ff_safe_queue_push_back(gg_model->request_queue, request) < 0) {
        destroy_request_item(&request);
        av_log(ctx, AV_LOG_ERROR, "Failed to push back request_queue.\n");
    }
}

static int execute_model_gg(GGRequestItem *request, Queue *lltask_queue)
{
    GGModel *gg_model;
    DnnContext *ctx;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    int ret = 0;

    if (ff_queue_size(lltask_queue) == 0) {
        destroy_request_item(&request);
        return 0;
    }

    lltask = ff_queue_peek_front(lltask_queue);
    task = lltask->task;
    gg_model = task->model;
    ctx = gg_model->ctx;

    ret = fill_model_input_gg(gg_model, request);
    if (ret != 0) {
        goto err;
    }

    if (task->async) {
        if (ff_dnn_start_inference_async(ctx, &request->exec_module) != 0) {
            goto err;
        }
        return 0;
    }
    else {
        ret = gg_start_inference(request);
        if (ret != 0) {
            goto err;
        }
        infer_completion_callback(request);
        return (task->inference_done == task->inference_todo) ? 0 : DNN_GENERIC_ERROR;
    }
err:
    gg_free_request(request->infer_request);
    if (ff_safe_queue_push_back(gg_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }

    return ret;
}

static int dnn_execute_model_gg(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    GGModel *gg_model = (GGModel *)model;
    DnnContext *ctx = gg_model->ctx;
    TaskItem *task;
    GGRequestItem *request;
    int ret = 0;

    ret = ff_check_exec_params(ctx, DNN_GG, model->func_type, exec_params);
    if (ret != 0) {
        return ret;
    }

    task = av_malloc(sizeof(*task));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return AVERROR(ENOMEM);
    }

    ret = ff_dnn_fill_task(task, exec_params, gg_model, ctx->async, 1);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Fill task with invalid parameter(s).\n");
        av_freep(&task);
        return ret;
    }

    if (ff_queue_push_back(gg_model->task_queue, task) < 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        return AVERROR(ENOMEM);
    }

    ret = extract_lltask_from_task(task, gg_model->lltask_queue);
    if (ret != 0) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        return ret;
    }

    request = ff_safe_queue_pop_front(gg_model->request_queue);
    if (!request) {
        av_freep(&task);
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }
    return execute_model_gg(request, gg_model->lltask_queue);
}

static DNNAsyncStatusType dnn_get_result_gg(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    GGModel *gg_model = (GGModel *)model;
    return ff_dnn_get_result_common(gg_model->task_queue, in, out);
}

static int dnn_flush_gg(const DNNModel *model)
{
    GGModel *gg_model = (GGModel *)model;
    DnnContext *ctx = gg_model->ctx;
    GGRequestItem *request;
    int ret;

    if (ff_queue_size(gg_model->lltask_queue) == 0) {
        // no pending task need to flush
        return 0;
    }

    request = ff_safe_queue_pop_front(gg_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    ret = fill_model_input_gg(gg_model, request);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to fill model input.\n");
        if (ff_safe_queue_push_back(gg_model->request_queue, request) < 0) {
            destroy_request_item(&request);
        }
        return ret;
    }

    return ff_dnn_start_inference_async(ctx, &request->exec_module);
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
