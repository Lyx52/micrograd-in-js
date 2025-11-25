import type {ICallable} from "../interfaces/ICallable.ts";
import {NewTensor} from "../tensor_new.ts";
import type {IParameterized, ParameterUpdateCallback} from "../interfaces/IParameterized.ts";

export class Neuron implements ICallable, IParameterized {
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

        return sum.tanh();
    }

    public zerograd() {
        this._weights.zerograd();

        if (this._useBias) {
            this._bias.zerograd();
        }
    }
}