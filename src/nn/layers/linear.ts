import type {ICallable} from "../interfaces/ICallable.ts";
import type {Value} from "../value.ts";
import {Tensor} from "../tensor.ts";
import type {IParameterized, ParameterUpdateCallback} from "../interfaces/IParameterized.ts";
import {NewTensor} from "../tensor_new.ts";

class LinearNeuron implements ICallable, IParameterized {
    private _weights: NewTensor;
    private _bias: NewTensor;
    private _useBias = true;

    constructor(inputs: number, useBias = true) {
        this._weights = NewTensor.randn(inputs);
        this._useBias = useBias;
        if (this._useBias) {
            this._bias = NewTensor.randn(1);
        }
    }

    public updateParameters(update: ParameterUpdateCallback) {
        update(this._weights);

        if (this._useBias) {
            update(this._bias);
        }
    }

    public execute(inputs: NewTensor): NewTensor {
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

        return sum;
    }

    public zerograd() {
        this._weights.zerograd();

        if (this._useBias) {
            this._bias.zerograd();
        }
    }
}

export class Linear implements ICallable, IParameterized {
    private _neurons: LinearNeuron[] = [];
    constructor(inputs: number, outputs: number, useBias = true) {
        for (let i = 0; i < outputs; i++) {
            this._neurons.push(new LinearNeuron(inputs, useBias))
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