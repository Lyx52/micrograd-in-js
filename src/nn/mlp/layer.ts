import type {ICallable} from "../interfaces/ICallable.ts";
import { Neuron } from "./neuron.ts";
import type {IParameterized, ParameterUpdateCallback} from "../interfaces/IParameterized.ts";
import {NewTensor} from "../tensor_new.ts";

export class Layer implements ICallable, IParameterized {
    private _neurons: Neuron[] = [];

    constructor(inputs: number, outputs: number, useBias = true) {
        for (let i = 0; i < outputs; i++) {
            this._neurons.push(new Neuron(inputs, useBias))
        }
    }

    public updateParameters(update: ParameterUpdateCallback) {
        for (const neuron of this._neurons) {
            neuron.updateParameters(update);
        }
    }

    public zerograd() {
        for (const neuron of this._neurons) {
            neuron.zerograd();
        }
    }

    public execute(inputs: NewTensor): NewTensor {
        const results: NewTensor[] = [];

        for (const neuron of this._neurons) {
            results.push(neuron.execute(inputs));
        }

        return NewTensor.fromTensors(results);
    }
}