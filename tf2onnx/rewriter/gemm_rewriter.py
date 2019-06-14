# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewrite - rewrite tensorflow subgraph to onnx gemm op
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=missing-docstring

def rewrite_gemm(g, ops):
    if g.opset < 6:
        return ops

    # some potential patterns for match
    pattern0 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('MatMul', name='matmul', inputs=[
                    OpTypePattern('*', name='A'),
                    OpTypePattern('*', name='B'),
                ]),
                OpTypePattern('Const',name='alpha')
            ]),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('*', name='C'),
                OpTypePattern('Const', name='beta')
            ]),
        ])

    # there is no beta
    pattern1 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('MatMul', name='matmul', inputs=[
                    OpTypePattern('*', name='A'),
                    OpTypePattern('*', name='B'),
                ]),
                OpTypePattern('Const', name='alpha')
            ]),
            OpTypePattern('*', name='C'),
        ])

    # there is no alpha
    pattern2 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul', inputs=[
                OpTypePattern('*', name='A'),
                OpTypePattern('*', name='B'),
            ]),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('*', name='C'),
                OpTypePattern('Const', name='beta')
            ]),
        ])

    # there are no beta and alpha
    pattern3 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul', inputs=[
                OpTypePattern('*', name='A'),
                OpTypePattern('*', name='B'),
            ]),
            OpTypePattern('*', name='C'),
        ])

    patternList = [pattern0, pattern1, pattern2, pattern3]   #append new pattern

    patternID = 1000
    for i, tempPattern in enumerate(patternList):
        matcher = GraphMatcher(tempPattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        if len(match_results)>0:
            patternID = i
            break

    if patternID==1000:
        return ops
    # print('patternID = ')
    # print(patternID)

    if patternID==0:    #there are both alpha and beta
        for match in match_results:
            # output nodes:
            add_node = match.get_op('add')

            matmul_node = match.get_op("matmul")
            mul2_node = match.get_op("mul2")
            mul1_node = match.get_op("mul1")

            inputA_node = match.get_op("A")
            inputB_node = match.get_op("B")
            inputC_node = match.get_op("C")

            # for edges:
            a_edge_name = _find_edges_name_btw_nodes(inputA_node, matmul_node)
            b_edge_name = _find_edges_name_btw_nodes(inputB_node, matmul_node)
            c_edge_name = _find_edges_name_btw_nodes(inputC_node, mul2_node)

            alpha = match.get_op("alpha").get_tensor_value()
            beta = match.get_op("beta").get_tensor_value()

            gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name], attr={"alpha": alpha, "beta": beta},
                               shapes=[g.get_shape(add_node.output[0])], dtypes=[g.get_dtype(add_node.output[0])]) # add_node.output_shapes[0]

            ops.remove(add_node)
            ops.remove(matmul_node)
            ops.remove(mul1_node)
            ops.remove(mul2_node)

    if patternID==1:    # there is no beta
        for match in match_results:
            # output nodes:
            add_node = match.get_op('add')

            matmul_node = match.get_op("matmul")
            #there is no mul2 node
            mul1_node = match.get_op("mul1")

            inputA_node = match.get_op("A")
            inputB_node = match.get_op("B")
            inputC_node = match.get_op("C")

            # for edges:
            a_edge_name = _find_edges_name_btw_nodes(inputA_node, matmul_node)
            b_edge_name = _find_edges_name_btw_nodes(inputB_node, matmul_node)
            c_edge_name = _find_edges_name_btw_nodes(inputC_node, add_node)

            alpha = match.get_op("alpha").get_tensor_value()

            gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name],
                               attr={"alpha": alpha},
                               shapes=[g.get_shape(add_node.output[0])],
                               dtypes=[g.get_dtype(add_node.output[0])])  # add_node.output_shapes[0]

            ops.remove(add_node)
            ops.remove(matmul_node)
            ops.remove(mul1_node)

    if patternID==2:    #there is no alpha
        for match in match_results:
            # output nodes:
            add_node = match.get_op('add')

            matmul_node = match.get_op("matmul")
            mul2_node = match.get_op("mul2")
            # there is no mul1

            inputA_node = match.get_op("A")
            inputB_node = match.get_op("B")
            inputC_node = match.get_op("C")

            # for edges:
            a_edge_name = _find_edges_name_btw_nodes(inputA_node, matmul_node)
            b_edge_name = _find_edges_name_btw_nodes(inputB_node, matmul_node)
            c_edge_name = _find_edges_name_btw_nodes(inputC_node, mul2_node)

            beta = match.get_op("beta").get_tensor_value()

            gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name], attr={"beta": beta},
                               shapes=[g.get_shape(add_node.output[0])], dtypes=[g.get_dtype(add_node.output[0])]) # add_node.output_shapes[0]

            ops.remove(add_node)
            ops.remove(matmul_node)
            ops.remove(mul2_node)

    if patternID==3:    #there are no alpha and beta
        for match in match_results:
            # output nodes:
            add_node = match.get_op('add')

            matmul_node = match.get_op("matmul")
            # there are no mul1 and mul2

            inputA_node = match.get_op("A")
            inputB_node = match.get_op("B")
            inputC_node = match.get_op("C")

            # for edges:
            a_edge_name = _find_edges_name_btw_nodes(inputA_node, matmul_node)
            b_edge_name = _find_edges_name_btw_nodes(inputB_node, matmul_node)
            c_edge_name = _find_edges_name_btw_nodes(inputC_node, add_node)

            # alpha = match.get_op("alpha").get_tensor_value()
            # beta = match.get_op("beta").get_tensor_value()

            gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name],
                               shapes=[g.get_shape(add_node.output[0])], dtypes=[g.get_dtype(add_node.output[0])]) # add_node.output_shapes[0]

            ops.remove(add_node)
            ops.remove(matmul_node)

    ops.append(gemm)
    g.replace_all_inputs(ops, add_node.output[0], gemm.output[0])

    return ops

def _find_edges_name_btw_nodes(sender, sinker):
    for sinker_end in sinker.input:
        for sender_end in sender.output:
            if sinker_end == sender_end:
                return sinker_end
    return None