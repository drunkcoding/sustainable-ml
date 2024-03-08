"""
Usage:
python3 -m unittest -bv test_solver_mlp.py
"""
from collections import defaultdict
from enum import Enum
import unittest

import numpy as np

from hlo import *
from cluster_env import ClusterEnvironment
from solver import solve_auto_sharding, SolverOption

MB = 1024 ** 2


def assert_close(x, y):
    assert abs(x / y - 1) < 0.001, f"{x} vs. {y}"


def solve_without_all_gather(computation, mesh_shape):
    device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
    solver_option = SolverOption()
    solver_option.force_all_gather_cost = 1e8
    cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                     memory_per_device=1000 * MB,
                                     solver_option=solver_option)
    objective = solve_auto_sharding(computation, cluster_env, solver_option)
    return objective, cluster_env


def get_mlp_2_layer_computation(batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))

        ## forward
        h1 = HloDot(x, w1)
        h2 = HloDot(h1, w2)
        loss = HloSubtract(h2, y)

        ## backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)

        grad_w2 = HloDot(h1, grad_loss,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w2 = HloSubtract(w2, grad_w2)
        grad_h1 = HloDot(grad_loss, w2,
                         lhs_contracting_dims=(1,),
                         rhs_contracting_dims=(1,),)

        grad_w1 = HloDot(x, grad_h1,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w1 = HloSubtract(w1, grad_w1)
        out = HloTuple((new_w1, new_w2))

        ## alias
        computation.set_alias([(w1, new_w1), (w2, new_w2)])

        """
         0: parameter.0 (128, 1024) = parameter()
         1: parameter.1 (128, 1024) = parameter()
         2: parameter.2 (1024, 1024) = parameter()
         3: parameter.3 (1024, 1024) = parameter()
         4: dot.0 (128, 1024) = dot(parameter.0, parameter.2)  lhs_con_dim=(1,), rhs_con_dim=(0,)
         5: dot.1 (128, 1024) = dot(dot.0, parameter.3)  lhs_con_dim=(1,), rhs_con_dim=(0,)
         6: subtract.0 (128, 1024) = subtract(dot.1, parameter.1)
         7: constant.0 () = constant(1.52587891e-05)
         8: broadcast.0 (128, 1024) = broadcast(constant.0)
         9: multiply.0 (128, 1024) = multiply(subtract.0, broadcast.0)
        10: dot.2 (1024, 1024) = dot(dot.0, multiply.0)  lhs_con_dim=(0,), rhs_con_dim=(0,)
        11: subtract.1 (1024, 1024) = subtract(parameter.2, dot.2)
        12: dot.3 (128, 1024) = dot(multiply.0, parameter.3)  lhs_con_dim=(1,), rhs_con_dim=(1,)
        13: dot.4 (1024, 1024) = dot(parameter.0, dot.3)  lhs_con_dim=(0,), rhs_con_dim=(0,)
        14: subtract.2 (1024, 1024) = subtract(parameter.2, dot.4)
        15: tuple.0 () = tuple('subtract.2', 'subtract.1') 
        """
    return computation


def get_mlp_2_layer_bias_computation(batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w1 = HloParameter((input_dim, hidden_dim))
        w2 = HloParameter((hidden_dim, output_dim))
        b1 = HloParameter((hidden_dim,))
        b2 = HloParameter((output_dim,))

        ## forward
        h1 = HloDot(x, w1)
        bb1 = HloBroadcast(b1, (batch_size, hidden_dim), dimensions=(1,))
        h1_add = HloAdd(h1, bb1)

        h2 = HloDot(h1_add, w2)
        bb2 = HloBroadcast(b2, (batch_size, output_dim), dimensions=(1,))
        h2_add = HloAdd(h2, bb2)

        loss = HloSubtract(h2_add, y)

        ## backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)

        grad_w2 = HloDot(h1_add, grad_loss,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w2 = HloSubtract(w2, grad_w2)

        grad_h1 = HloDot(grad_loss, w2,
                         lhs_contracting_dims=(1,),
                         rhs_contracting_dims=(1,),)

        grad_w1 = HloDot(x, grad_h1,
                         lhs_contracting_dims=(0,),
                         rhs_contracting_dims=(0,),)
        new_w1 = HloSubtract(w1, grad_w1)

        grad_b1 = HloReduce(grad_h1, dimensions=[0])
        new_b1 = HloSubtract(b1, grad_b1)

        grad_b2 = HloReduce(grad_loss, dimensions=[0])
        new_b2 = HloSubtract(b2, grad_b2)

        out = HloTuple((new_w1, new_w2, new_b1, new_b2))

        ## alias
        computation.set_alias([(w1, new_w1), (w2, new_w2)])

    return computation


def get_mlp_n_layer_computation(num_layers, batch_size, input_dim, hidden_dim, output_dim):
    computation = HloComputation()
    with computation:
        x = HloParameter((batch_size, input_dim))
        y = HloParameter((batch_size, output_dim))
        w_first = HloParameter((input_dim, hidden_dim))
        w_inter = []
        for i in range(num_layers - 2):
            manual_strategy = "S0" if i % 2 == 0 else "S1"
            w_inter.append(HloParameter((hidden_dim, hidden_dim)))
        w_last = HloParameter((hidden_dim, output_dim))

        # forward
        h_first = HloDot(x, w_first)
        h_now = h_first
        h_inter = []
        for i in range(num_layers - 2):
            h_now = HloDot(h_now, w_inter[i])
            h_inter.append(h_now)
        h_last = HloDot(h_now, w_last)

        loss = HloSubtract(h_last, y)

        # backward
        coef = HloConstant(2 / batch_size / output_dim)
        coef = HloBroadcast(coef, (batch_size, output_dim))
        grad_loss = HloMutiply(loss, coef)
        grad_h_now = grad_loss

        grad_w_last = HloDot(h_inter[-1], grad_h_now,
                             lhs_contracting_dims=(0,),
                             rhs_contracting_dims=(0,),)
        new_w_last = HloSubtract(w_last, grad_w_last)
        grad_h_now = HloDot(grad_h_now, w_last,
                             lhs_contracting_dims=(1,),
                             rhs_contracting_dims=(1,),)

        new_w_inter = []
        for i in range(num_layers - 3, -1, -1):
            grad_w = HloDot(h_inter[i-1], grad_h_now,
                            lhs_contracting_dims=(0,),
                            rhs_contracting_dims=(0,),)
            new_w = HloSubtract(w_inter[i], grad_w)
            grad_h_now = HloDot(grad_h_now, w_inter[i],
                                lhs_contracting_dims=(1,),
                                rhs_contracting_dims=(1,),)
            new_w_inter.append(new_w)

        grad_w_first = HloDot(x, grad_h_now,
                              lhs_contracting_dims=(0,),
                              rhs_contracting_dims=(0,),)
        new_w_first = HloSubtract(w_first, grad_w_first)

        out = HloTuple([new_w_first] + new_w_inter + [new_w_last])

        # alias
        alias_list = [(w_first, new_w_first), (w_last, new_w_last)] +\
            [(w_old, w_new) for w_old, w_new in zip(w_inter, reversed(new_w_inter))]
        computation.set_alias(alias_list)
    return computation


def get_attention_forward_computation(batch_size, seq_len, hidden_dim, num_head, force_replicated_output):
    per_head = hidden_dim // num_head
    computation = HloComputation()

    with computation:
        # hidden states
        hidden_states = HloParameter((batch_size, seq_len, hidden_dim))
        hidden_states = HloReshape(hidden_states, (batch_size * seq_len, hidden_dim))

        # query matmul
        weight_query_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_query_dense_ = HloReshape(weight_query_dense, (hidden_dim, hidden_dim))
        query = HloDot(hidden_states, weight_query_dense_)
        query = HloReshape(query, (batch_size, seq_len, num_head, per_head))

        # query bias_add
        bias_query_dense = HloParameter((num_head, per_head))
        bias_query_dense_ = HloBroadcast(bias_query_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        query = HloAdd(query, bias_query_dense_)

        # query normalization
        c = HloConstant(0.125)
        c = HloBroadcast(c, (batch_size, seq_len, num_head, per_head))
        query = HloMutiply(c, query)
        # query transpose
        query = HloTranspose(query, [0, 2, 1, 3])

        # key matmul
        weight_key_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_key_dense_ = HloReshape(weight_key_dense, (hidden_dim, hidden_dim))
        key = HloDot(hidden_states, weight_key_dense_)
        key = HloReshape(key, (batch_size, seq_len, num_head, per_head))

        # key bias_add
        bias_key_dense = HloParameter((num_head, per_head))
        bias_key_dense_ = HloBroadcast(bias_key_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        key = HloAdd(key, bias_key_dense_)

        # key transpose
        key = HloTranspose(key, [0, 2, 3, 1])

        # att_weight
        att_weight = HloDot(query, key,
                            lhs_batch_dims=(0,1), lhs_contracting_dims=(3,),
                            rhs_batch_dims=(0,1), rhs_contracting_dims=(2,))

        # mask
        mask = HloParameter((batch_size, seq_len))

        # attention_bias_pred
        zero = HloConstant(0)
        zero = HloBroadcast(zero, (batch_size, seq_len))
        pred = HloCompare(mask, zero)

        # all zero
        zero = HloConstant(0)
        zero = HloBroadcast(zero, (batch_size, seq_len))

        # all neg-infinity
        neg_inf = HloConstant(-1e10)
        neg_inf = HloBroadcast(neg_inf, (batch_size, seq_len))

        # attention bias
        select = HloSelect(pred, zero, neg_inf)

        # attention bias_add
        att_bias = HloBroadcast(select, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 3))
        att_weight = HloAdd(att_weight, att_bias)

        # softmax_max
        max_reduce = HloReduce(att_weight, dimensions=(3,))
        max_reduce = HloBroadcast(max_reduce, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 1, 2))
        diff = HloSubtract(att_weight, max_reduce)
        exp = HloExp(diff)
        # softmax_sum
        sum_reduce = HloReduce(exp, dimensions=(3,))
        sum_reduce = HloBroadcast(sum_reduce, (batch_size, num_head, seq_len, seq_len), dimensions=(0, 1, 2))
        # softmax_norm
        softmax = HloDiv(exp, sum_reduce)

        # value matmul
        weight_value_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_value_dense_ = HloReshape(weight_value_dense, (hidden_dim, hidden_dim))
        value = HloDot(hidden_states, weight_value_dense_)
        value = HloReshape(value, (batch_size, seq_len, num_head, per_head))

        # value bias_add
        bias_value_dense = HloParameter((num_head, per_head))
        bias_value_dense_ = HloBroadcast(bias_value_dense, (batch_size, seq_len, num_head, per_head), dimensions=(2, 3))
        value = HloAdd(value, bias_value_dense_)

        # value transpose
        value = HloTranspose(value, [0, 2, 3, 1])

        # self attention
        self_att = HloDot(value, softmax,
                          lhs_batch_dims=(0, 1), lhs_contracting_dims=(3,),
                          rhs_batch_dims=(0, 1), rhs_contracting_dims=(3,))
        self_att = HloTranspose(self_att, [0, 3, 1, 2])
        self_att = HloReshape(self_att, [batch_size * seq_len, hidden_dim])

        # out matmul
        weight_out_dense = HloParameter((hidden_dim, num_head, per_head))
        weight_out_dense_ = HloReshape(weight_out_dense, (hidden_dim, hidden_dim))
        out = HloDot(self_att, weight_out_dense_)
        out = HloReshape(out, (batch_size, seq_len, hidden_dim))

        # out bias_add
        bias_out_dense = HloParameter((hidden_dim,))
        bias_out_dense_ = HloBroadcast(bias_out_dense, (batch_size, seq_len, hidden_dim), dimensions=(2,))
        out = HloAdd(out, bias_out_dense_)

        if force_replicated_output:
            out = HloForceReplicated(out)

        out = HloTuple([out,
                        weight_value_dense, bias_value_dense,
                        weight_query_dense, bias_query_dense,
                        weight_key_dense, bias_key_dense,
                        weight_out_dense, bias_out_dense,
        ])

    return computation

def mlp_2_layer(batch_size=1024, hidden_dim=128):
    computation = get_mlp_2_layer_computation(batch_size, hidden_dim,hidden_dim, hidden_dim)

    # Test different device meshes
    for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
        device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
        cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
        objective = solve_auto_sharding(computation, cluster_env)
    return objective

def attention_forward(batch_size=4, seq_len=128, hidden_dim=512, num_head=16):
    computation = get_attention_forward_computation(
            batch_size, seq_len, hidden_dim, num_head, True)

    # Solve
    for i, mesh_shape in enumerate([ (4, 1), (1, 4) ]):
        objective, cluster_env = solve_without_all_gather(computation, mesh_shape)
    return objective
    
def mlp_2_layer_bias(batch_size=128, hidden_dim=1024):
    computation = get_mlp_2_layer_bias_computation(batch_size, hidden_dim,
            hidden_dim, hidden_dim)

        # Test different device meshes
    for i, mesh_shape in enumerate([(4, 1), (1, 4)]):
        device_mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
        cluster_env = ClusterEnvironment(device_mesh, [1, 1], [1, 1],
                                             memory_per_device=1000 * MB)
        objective = solve_auto_sharding(computation, cluster_env)
    return objective



