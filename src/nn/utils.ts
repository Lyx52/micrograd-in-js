import { Value } from "./value";
import type {ICallable} from "./ICallable.ts";
import {Network} from "vis-network";
import {v4 as uuid} from "uuid";
import { Tensor } from "./tensor.ts";
export const mse = (expected: number, result: number|Value): number => {
    return result instanceof Value ? Math.pow(expected - result.Data, 2) : Math.pow(expected - result, 2);
}

export const getTotalElements = (...dims: number[]): number => {
    let elements = dims[0];
    for (let i = 0; i < dims.length - 1; i++) {
        elements *= dims[i + 1];
    }

    return elements;
}

export const crossEntropyLoss = (network: ICallable, ys: Tensor[], xs: Tensor[]) => {
    const losses: Tensor[] = [];
    for (let i = 0; i < xs.length; i++) {
        const result = network.execute(xs[i]);
        losses.push(result.mse(ys[i]));
    }

    return Tensor.fromTensors(...losses).sum();
}

export const renderGraph = (root: Value) => {
    const [nodes, edges] = root.graph()

    const data = {
        nodes: nodes,
        edges: edges,
    };

    const options = {
        "edges": {
            "smooth": {
                "type": "vertical"
            }
        },
        "physics": {
            "barnesHut": {
                "theta": 0.4,
                "gravitationalConstant": -31320,
                "centralGravity": 0.85,
                "springLength": 175,
                "avoidOverlap": 0.31
            },
            "minVelocity": 0.75
        }
    }
    //document.querySelector('.graph-container')?.remove();
    const container = document.createElement('div');
    container.id = uuid();
    container.style.width = '100%';
    container.style.height = '95vh';
    container.classList.add('graph-container');
    document.body.appendChild(container);
    const network = new Network(container, data, options as any);
}