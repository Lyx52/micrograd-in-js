import { Value } from "./value";
import type {ICallable} from "./interfaces/ICallable.ts";
import {Network} from "vis-network";
import {v4 as uuid} from "uuid";
import {NewTensor} from "./tensor_new.ts";
export const mse = (expected: number, result: number|Value): number => {
    return result instanceof Value ? Math.pow(expected - result.Data, 2) : Math.pow(expected - result, 2);
}
const BENCHMARKS_ENABLED = false;
export const benchmarkStart = (name: string) => {
    if (BENCHMARKS_ENABLED) {
        console.time(name)
    }
};
export const benchmarkEnd = (name: string) => {
    if (BENCHMARKS_ENABLED) {
        console.timeEnd(name)
    }
};

export const getTotalElements = (...dims: number[]): number => {
    let elements = dims[0];
    for (let i = 0; i < dims.length - 1; i++) {
        elements *= dims[i + 1];
    }

    return elements;
}

export const crossEntropyLoss = (network: ICallable, ys: NewTensor[], xs: NewTensor[]) => {
    const losses: NewTensor[] = [];
    for (let i = 0; i < xs.length; i++) {
        const result = network.execute(xs[i]);
        losses.push(result.mse(ys[i]));
    }

    return NewTensor.fromTensors(losses).sum();
}

const sampleRandom = (count: number, ys: NewTensor[], xs: NewTensor[]) => {
    const indexes = [];
    for (let i = 0; i < count; i++) {
        indexes.push(Math.floor(Math.random() * xs.length));
    }

    return [
        indexes.map(i => ys[i].clone()),
        indexes.map(i => xs[i])
    ];
}

export const maxMarginLoss = (network: ICallable, ys: NewTensor[], xs: NewTensor[], batchSize: number = 10) => {
    const [ydata, xdata] = sampleRandom(batchSize, ys, xs);
    let relu = undefined;
    for (let i = 0; i < xdata.length; i++) {
        const result = network.execute(xdata[i]);
        relu = ydata[i].mul(result).negate().add(1).relu();
    }

    return NewTensor.from([1])
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