import {Value} from "./value.ts";
import type {ICallable} from "./ICallable.ts";
import {v4 as uuid} from "uuid";
import {Tensor} from "./tensor.ts";

export class Neuron implements ICallable {
    private _weights: Tensor;
    private _bias: Tensor;
    private _id = null;
    constructor(inputs: number) {
        this._id = uuid();
        this._bias = Tensor.randn(1);
        this._weights = Tensor.randn(inputs);
    }

    execute(inputs: Tensor): Tensor {
        // tanh of (sum of (w * x) + bias)
        if (!this._weights.equalDimensions(inputs)) {
            throw new RangeError("Tensors must be equal");
        }

        return this._weights
            .mul(inputs)
            .sum()
            .add(this._bias)
            .tanh();
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