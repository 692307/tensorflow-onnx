# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewrite - rewrite tensorflow subgraph to onnx gemm op
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=missing-docstring

def rewrite_gemm(g, ops):
    if g.opset <= 6:
        return ops

    """
    4 Candidate patterns are listed as follow, i.e. pattern0, pattern1, pattern2 and pattern 3
    Where, A,B and C represent the three inputs, alpha and beta represent the two attributes.
    """
    # pattern0: alpha*A*B + beta*C
    pattern0 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('Const', name='alpha'),
                OpTypePattern('MatMul', name='matmul', inputs=[
                    OpTypePattern('*', name='A'),
                    OpTypePattern('*', name='B'),
                ]),
            ]),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C'),
            ])
        ])


    # pattern1: alpha*A*B + C
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

    # pattern2: A*B + beta*C
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

    # pattern3: A*B + C
    pattern3 = \
        OpTypePattern('Add', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul', inputs=[
                OpTypePattern('*', name='A'),
                OpTypePattern('*', name='B'),
            ]),
            OpTypePattern('*', name='C'),
        ])

    patternList = [pattern0, pattern1, pattern2, pattern3]

    for patternID, tempPattern in enumerate(patternList):
        matcher = GraphMatcher(tempPattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        if len(match_results)>0:
            print(patternID,patternID)
            for match in match_results:
                add_node = match.get_op('add')
                matmul_node = match.get_op("matmul")
                inputA_node = match.get_op("A")
                inputB_node = match.get_op("B")
                inputC_node = match.get_op("C")
                a_edge_name = inputA_node.output[0]
                b_edge_name = inputB_node.output[0]
                c_edge_name = inputC_node.output[0]

                attr4makenode = {}

                if patternID == 0:  # pattern0: alpha*A*B + beta*C
                    mul2_node = match.get_op("mul2")
                    mul1_node = match.get_op("mul1")
                    alpha = match.get_op("alpha").get_tensor_value()
                    beta = match.get_op("beta").get_tensor_value()
                    attr4makenode = {"alpha": alpha, "beta": beta}

                if patternID == 1:  # pattern1: alpha*A*B + C
                    mul1_node = match.get_op("mul1")
                    alpha = match.get_op("alpha").get_tensor_value()
                    attr4makenode = {"alpha": alpha}

                if patternID == 2:  # pattern2: A*B + beta*C
                    mul2_node = match.get_op("mul2")
                    beta = match.get_op("beta").get_tensor_value()
                    attr4makenode = attr={"beta": beta}

                # if the pattern is 3, do nothing
                gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name],
                                   attr=attr4makenode,
                                   # outputs=[add_node.output[0]],
                                   shapes=[g.get_shape(add_node.output[0])],
                                   dtypes=[g.get_dtype(add_node.output[0])])
                ops.remove(add_node)
                ops.remove(matmul_node)
                if patternID == 0 or patternID == 1:
                    ops.remove(mul1_node)
                if patternID == 0 or patternID == 2:
                    ops.remove(mul2_node)

            ops.append(gemm)
            g.replace_all_inputs(ops, add_node.output[0], gemm.output[0])

    return ops