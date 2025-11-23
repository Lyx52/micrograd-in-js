import {Value} from "./value.ts";
import type {ICallable} from "./ICallable.ts";
import {Tensor} from "./tensor.ts";

export class Neuron implements ICallable {
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

    execute(inputs: Tensor): Tensor {
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
        for (const parameter of this.parameters()) {
            parameter.zerograd();
        }
    }

    parameters(): Value[] {
        return [...this._weights.item(), ...this._bias.item()];
    }
}