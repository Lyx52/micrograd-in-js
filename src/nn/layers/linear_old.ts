import type {ICallable} from "../interfaces/ICallable.ts";
import type {Value} from "../../old/value.ts";
import {Tensor} from "../../old/tensor.ts";
import type {IParameterized, ParameterUpdateCallback} from "../interfaces/IParameterized.ts";
import {NewTensor} from "../tensor_new.ts";

class LinearNeuronOld {
    private _weights: Tensor;
    private _bias: Tensor;
    private _useBias = true;

    constructor(inputs: number, useBias = true) {
        this._weights = Tensor.randn(inputs);
        this._useBias = useBias;
        if (this._useBias) {
            this._bias = Tensor.randn(1);
        }
    }

    public execute(inputs: Tensor): Tensor {
        // tanh of (sum of (w * x) + bias)
        if (!this._weights.equalDimensions(inputs)) {
            throw new RangeError("Tensors must be equal");
        }
        let sum = this._weights
            .mul(inputs)
            .sum();

        if (this._useBias) {
            sum = sum.add(this._bias);
        }

        return sum.tanh();
    }
}

export class LinearOld {
    private _neurons: LinearNeuronOld[] = [];
    constructor(inputs: number, outputs: number, useBias = true) {
        for (let i = 0; i < outputs; i++) {
            this._neurons.push(new LinearNeuronOld(inputs, useBias))
        }
    }
    public execute(inputs: Tensor): Tensor {
        const results: Tensor[] = [];

        for (const neuron of this._neurons) {
            results.push(neuron.execute(inputs));
        }

        return Tensor.fromTensors(...results);
    }
}