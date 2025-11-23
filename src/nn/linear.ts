import type {ICallable} from "./ICallable.ts";
import type {Value} from "./value.ts";
import {Tensor} from "./tensor.ts";
import type {IParameterized} from "./IParameterized.ts";

class LinearNeuron implements ICallable, IParameterized {
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

        return sum;
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

export class Linear implements ICallable, IParameterized {
    private _neurons: LinearNeuron[] = [];
    constructor(inputs: number, outputs: number, useBias = true) {
        for (let i = 0; i < outputs; i++) {
            this._neurons.push(new LinearNeuron(inputs, useBias))
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