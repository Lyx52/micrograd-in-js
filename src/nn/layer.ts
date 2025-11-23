import type {ICallable} from "./ICallable.ts";
import { Neuron } from "./neuron.ts";
import type {Value} from "./value.ts";
import {Tensor} from "./tensor.ts";
import type {IParameterized} from "./IParameterized.ts";

export class Layer implements ICallable, IParameterized {
    private _neurons: Neuron[] = [];
    constructor(inputs: number, outputs: number, useBias = true) {
        for (let i = 0; i < outputs; i++) {
            this._neurons.push(new Neuron(inputs, useBias))
        }
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