import type {ICallable} from "./ICallable.ts";
import { Neuron } from "./neuron.ts";
import type {Value} from "./value.ts";
import {Tensor} from "./tensor.ts";

export class Layer implements ICallable {
    private _neurons: Neuron[] = [];
    private _inputs: number = 0;
    constructor(inputs: number, outputs: number) {
        for (let i = 0; i < outputs; i++) {
            this._neurons.push(new Neuron(inputs))
        }

        this._inputs = inputs;
    }

    zerograd() {
        for (const neuron of this._neurons) {
            neuron.zerograd();
        }
    }

    parameters(): Value[] {
        return this._neurons.flatMap(n => n.parameters());
    }

    execute(inputs: Tensor): Tensor {
        const results: Tensor[] = [];

        for (const neuron of this._neurons) {
            results.push(neuron.execute(inputs));
        }

        return Tensor.fromTensors(...results);
    }
}