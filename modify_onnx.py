import onnx_graphsurgeon as gs
import argparse
import onnx
import numpy as np
import json

def process_graph(graph):
    print(len(graph.inputs))
    tensors = graph.tensors()
    tensor = tensors['p2o.Concat.180']
    tensor.outputs.clear()
    graph.outputs[1] = tensor

    tensor_scores = tensors['sigmoid_32.tmp_0']
    tensor_scores.outputs.clear()
    graph.outputs[0] = tensor_scores

    graph.outputs[0].name = 'scores'
    graph.outputs[1].name = 'boxes'

    del graph.inputs[2]
    del graph.inputs[0]
    return graph


def main():
    parser = argparse.ArgumentParser(description="Modify DCNv2 plugin node into ONNX model")
    parser.add_argument("-i", "--input",
            help="Path to ONNX model with 'Plugin' node to replace with DCNv2_TRT",
            default="models/centertrack_DCNv2_named.onnx")
    parser.add_argument("-o", "--output",
            help="Path to output ONNX model with 'DCNv2_TRT' node",
            default="models/modified.onnx")

    args, _ = parser.parse_known_args()
    graph = gs.import_onnx(onnx.load(args.input))
    graph = process_graph(graph)

    # 删除现在游离的子图
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), args.output)

if __name__ == '__main__':
    main()